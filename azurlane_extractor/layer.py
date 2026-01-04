"""Game object layer handling and image compositing."""
import logging
from typing import Optional
from PIL import Image

from UnityPy.enums import ClassIDType

from .asset import AzurlaneAsset
from .constants import MESH_VR, MESH_TR, MESH_SR
from .config import get_config

log = logging.getLogger(__name__)


def parse(x):
    """Parse Unity vector/quaternion types to tuples."""
    if hasattr(x, "values"):
        return tuple(x.values())
    
    for attrs in [('X', 'Y', 'Z', 'W'), ('x', 'y', 'z', 'w')]:
        if hasattr(x, attrs[0]):
            result = [getattr(x, attrs[0])]
            for attr in attrs[1:]:
                if hasattr(x, attr):
                    result.append(getattr(x, attr))
                else:
                    break
            return tuple(result) if len(result) > 1 else result[0]
    return x


def recon(src: Image.Image, mesh: list[str]) -> Image.Image:
    """Reconstruct image from texture using mesh data."""
    sx, sy = src.size
    c = map(MESH_SR.split, list(filter(MESH_TR.match, mesh))[1::2])
    p = map(MESH_SR.split, list(filter(MESH_VR.match, mesh))[1::2])
    c = [(round(float(a[1]) * sx), round((1 - float(a[2])) * sy)) for a in c]
    p = [(-int(float(a[1])), int(float(a[2]))) for a in p]
    my = max(y for x, y in p)
    p = [(x, my - y) for x, y in p[::2]]
    cp = [(L + R, P) for L, R, P in zip(c[::2], c[1::2], p)]
    ox, oy = zip(*[(r - L + p, b - t + q) for (L, t, r, b), (p, q) in cp])
    out = Image.new('RGBA', (max(ox), max(oy)))
    for c, p in cp:
        out.paste(src.crop(c), p)
    return out


class RectTransform:
    """Parsed Unity RectTransform data."""
    
    def __init__(self, rtf):
        self.local_rotation = parse(rtf.m_LocalRotation)
        self.local_position = parse(rtf.m_LocalPosition)
        self.local_scale = parse(rtf.m_LocalScale)
        self.anchor_min = parse(rtf.m_AnchorMin)
        self.anchor_max = parse(rtf.m_AnchorMax)
        self.anchored_position = parse(rtf.m_AnchoredPosition)
        self.size_delta = parse(rtf.m_SizeDelta)
        self.pivot = parse(rtf.m_Pivot)
        
        # Extract Z position for depth sorting
        self.z_position = 0.0
        if isinstance(self.local_position, tuple) and len(self.local_position) >= 3:
            self.z_position = self.local_position[2]


class GameObjectLayer:
    """A layer in the painting hierarchy with image and transform data."""
    
    def __init__(self, asset: AzurlaneAsset, gameobject, parent: "GameObjectLayer" = None):
        self.asset = asset
        self.gameobject = gameobject
        self.parent = parent
        self.children: list[GameObjectLayer] = []
        self.sibling_index: int = -1  # Track position in parent's children list

        result = asset.getComponentFromObject(gameobject, types=[ClassIDType.RectTransform, ClassIDType.Transform])
        if result:
            self.transform, transform_type = result
            if transform_type == ClassIDType.RectTransform:
                self.rect_transform = RectTransform(self.transform)
        else:
            self.transform = None

        self.local_offset = (0, 0)
        self.global_offset = (0, 0)
        self.image: Optional[Image.Image] = None
        self._pending_scale_id: Optional[str] = None

        # Initialize size from rect_transform for layers without mesh (e.g., face layer)
        # This ensures correct offset calculation before image is loaded
        if hasattr(self, 'rect_transform'):
            self.size = self.rect_transform.size_delta
        else:
            self.size = (0, 0)

        self.meshimage = asset.getComponentFromObject(gameobject, types=[ClassIDType.MonoBehaviour], attributes={"mMesh"})
        if self.meshimage:
            self._loadImage(self.meshimage)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.gameobject.name}>"

    def _loadImage(self, meshimage):
        """Load and reconstruct image from mesh data."""
        from .scaler import add_to_batch_scaler
        config = get_config()
        
        try:
            self.sprite = meshimage.m_Sprite.read()
        except FileNotFoundError as e:
            log.warning(f"Skipping layer '{self.gameobject.m_Name}': sprite not found ({e})")
            return
        if not self.sprite:
            return
        
        self.texture2d = self.sprite.m_RD.texture.read()
        image = self.texture2d.image
        
        log.debug(f"LAYER '{self.gameobject.m_Name}': sprite='{self.sprite.m_Name}', "
              f"texture='{self.texture2d.m_Name}', size={image.size}")

        # Reconstruct from mesh if available
        self.mesh = self.asset.getObjectByPathID(meshimage.mMesh.path_id)
        if self.mesh:
            image = recon(image, self.mesh.export().splitlines())
        elif hasattr(meshimage.mMesh, 'read') and meshimage.mMesh.m_PathID != 0:
            self.mesh = meshimage.mMesh.read()
            if self.mesh:
                image = recon(image, self.mesh.export().splitlines())

        # Handle size adjustments between recon output and expected raw sprite size
        psizex, psizey = image.size
        pdeltax, pdeltay = self.rect_transform.size_delta
        prawx, prawy = parse(meshimage.mRawSpriteSize)

        if prawx != psizex or prawy != psizey:
            # Recon output doesn't match raw sprite size
            if psizex < prawx or psizey < prawy:
                # Recon output is smaller - paste onto canvas at correct position
                paste_x, paste_y = 0, 0
                if self.mesh:
                    mesh_lines = self.mesh.export().splitlines()
                    p = list(map(MESH_SR.split, list(filter(MESH_VR.match, mesh_lines))[1::2]))
                    if p:
                        vertices = [(-int(float(a[1])), int(float(a[2]))) for a in p]
                        max_y = max(y for x, y in vertices)
                        paste_y = int(prawy - max_y)  # Convert from Unity Y-up to screen Y-down
                
                empty_canvas = Image.new('RGBA', (int(prawx), int(prawy)), (0, 0, 0, 0))
                empty_canvas.paste(image, (paste_x, paste_y))
                image = empty_canvas

        # For positioning, always use size_delta (Unity's RectTransform size)
        # But the actual image can be larger (mesh reconstruction can extend beyond bounds)
        before_size = image.size
        target_size = (round(pdeltax), round(pdeltay))
        
        if config.external_scaler and before_size != target_size and before_size[0] <= target_size[0] and before_size[1] <= target_size[1]:
            # Only queue for external scaling if image needs to be scaled UP
            # Never scale down mesh-reconstructed images
            self._pending_scale_id = add_to_batch_scaler(image, target_size, self.gameobject.m_Name)
            self.image = image
        elif before_size != target_size and before_size[0] <= target_size[0] and before_size[1] <= target_size[1]:
            # Only scale UP to match size_delta, never scale DOWN
            self.image = image.resize(target_size, Image.LANCZOS)
            log.debug(f"SCALE Layer '{self.gameobject.m_Name}': {before_size} -> {target_size} (LANCZOS)")
        else:
            # Image is same size or larger - keep as-is
            self.image = image
            if before_size != target_size:
                log.debug(f"Keep mesh size '{self.gameobject.m_Name}': image={before_size}, size_delta={target_size}")
        
        # Always use size_delta for positioning calculations
        self.size = (pdeltax, pdeltay)

    def loadImageSimple(self, image: Image.Image):
        """Load a pre-processed image (for face overlays).
        
        Face images may have slightly different dimensions than size_delta.
        We use the actual image directly. The returned offset adjustment should
        be applied when compositing, not stored on the layer.
        
        Returns:
            Tuple of (offset_x_adj, offset_y_adj) to be added to global_offset when compositing
        """
        pdeltax, pdeltay = self.rect_transform.size_delta
        target_w, target_h = round(pdeltax), round(pdeltay)
        img_w, img_h = image.size
        
        # Use the actual face image directly
        self.image = image
        self.size = (img_w, img_h)
        
        # Return offset adjustment to center actual image within where size_delta would be
        # In PIL coordinates (Y down), if image is SHORTER than target, move UP (subtract Y)
        # If image is WIDER than target, move LEFT (subtract X)
        if (img_w, img_h) != (target_w, target_h):
            dx = (img_w - target_w) / 2
            dy = (img_h - target_h) / 2
            log.debug(f"FACE '{self.gameobject.m_Name}': img={image.size} vs target={target_w, target_h}, offset_adj=({-dx:.1f},{dy:.1f})")
            # X: negative dx for wider image (move left)
            # Y: positive dy for shorter image (move up, i.e., smaller Y in PIL coords)
            return (-dx, dy)
        return (0, 0)

    def retrieveChildren(self, recursive: bool = True):
        """Build child layer hierarchy."""
        if not self.transform:
            return
        for idx, child in enumerate(self.transform.m_Children):
            childtf = self.asset.getObjectByPathID(child.path_id)
            if childtf is None:
                continue
            childobject = self.asset.getObjectByPathID(childtf.m_GameObject.path_id)
            if childobject is None:
                continue
            objectlayer = GameObjectLayer(self.asset, childobject, self)
            objectlayer.sibling_index = idx  # Store position in children list
            if recursive:
                objectlayer.retrieveChildren(recursive)
            self.children.append(objectlayer)

    def findChildLayer(self, name: str) -> Optional["GameObjectLayer"]:
        """Find a child layer by name (recursive)."""
        for child in self.children:
            if child.gameobject.m_Name == name:
                return child
            found = child.findChildLayer(name)
            if found:
                return found
        return None

    def calculateLocalOffset(self, recursive: bool = True):
        """Calculate local offset relative to parent."""
        if hasattr(self, 'rect_transform'):
            anchor_min = self.rect_transform.anchor_min
            anchor_max = self.rect_transform.anchor_max
            anchorpos = self.rect_transform.anchored_position
            size_delta = self.rect_transform.size_delta
            pivot = self.rect_transform.pivot

            # Special fix for aotuo_3 face layer positioning issue (including variants like aotuo_3_hx)
            adjusted_anchorpos_y = anchorpos[1]
            if (self.gameobject.m_Name == 'face' and self.parent and 
                self.parent.gameobject.m_Name.startswith('aotuo_3')):
                # aotuo_3 has incorrect face anchored_position.y in Unity data
                # Correct value should be -190 instead of -60
                adjusted_anchorpos_y = -190.0

            if anchor_min == anchor_max:
                # Point anchor - element positioned relative to anchor point
                anchorposx = anchorpos[0] + (self.size[0] - size_delta[0]) * pivot[0]
                anchorposy = adjusted_anchorpos_y + (self.size[1] - size_delta[1]) * (pivot[1] - 1)
                adjusted_anchorpos = (anchorposx, anchorposy)

                # Use parent size for children, or own size for root
                ref_size = self.parent.size if self.parent else self.size
                offsetx = ref_size[0] * anchor_min[0] + adjusted_anchorpos[0] - pivot[0] * self.size[0]
                offsety = ref_size[1] * anchor_min[1] - adjusted_anchorpos[1] - (1 - pivot[1]) * self.size[1]
                self.local_offset = (offsetx, offsety)
            else:
                # Stretch anchor - element fills parent (anchor_min != anchor_max)
                sizex = (anchor_min[0] - anchor_max[0]) * self.parent.size[0]
                sizey = (anchor_min[1] - anchor_max[1]) * self.parent.size[1]
                self.size = (abs(sizex), abs(sizey))
                self.local_offset = (0, 0)

        if recursive:
            for child in self.children:
                child.calculateLocalOffset(recursive)

    def getSmallestOffset(self, parent_accumulated: tuple[float, float] = None) -> tuple[float, float]:
        """Get minimum accumulated offset across this layer and children to ensure all content fits on canvas.
        
        Args:
            parent_accumulated: Accumulated offset from parent layers
        """
        if parent_accumulated is None:
            parent_accumulated = (0.0, 0.0)
        
        # My accumulated offset
        my_accumulated = (
            parent_accumulated[0] + self.local_offset[0],
            parent_accumulated[1] + self.local_offset[1]
        )
        
        min_offx, min_offy = (float('inf'), float('inf'))
        
        if self.image:
            min_offx, min_offy = my_accumulated
            
        for child in self.children:
            offcx, offcy = child.getSmallestOffset(my_accumulated)
            if offcx != float('inf'):
                min_offx = min(min_offx, offcx)
            if offcy != float('inf'):
                min_offy = min(min_offy, offcy)
        
        # Fall back to (0, 0) only if no offsets found at all (root level)
        if parent_accumulated == (0.0, 0.0):
            if min_offx == float('inf'):
                min_offx = 0
            if min_offy == float('inf'):
                min_offy = 0
        return min_offx, min_offy
    
    def _hasImageChildren(self) -> bool:
        """Check if any descendants have images."""
        for child in self.children:
            if child.image or child._hasImageChildren():
                return True
        return False

    def calculateGlobalOffset(self, base_offset: tuple[float, float] = None, 
                               parent_accumulated: tuple[float, float] = None,
                               root_size: tuple[float, float] = None):
        """Calculate global offset for rendering.
        
        Args:
            base_offset: The root's local offset (used to position everything relative to root)
            parent_accumulated: Accumulated offset from all parent layers
            root_size: The root layer's size (for _bj alignment check)
        """
        # Always use the root's local_offset as the base offset for all layers
        root = None
        if base_offset is None:
            # Find the root
            root = self
            while root.parent:
                root = root.parent
            base_offset = root.local_offset
        if parent_accumulated is None:
            parent_accumulated = (0.0, 0.0)
        if root_size is None:
            root_size = self.size
        
        # Global offset = parent's accumulated offset + my local offset - base offset
        self.global_offset = (
            parent_accumulated[0] + self.local_offset[0] - base_offset[0],
            parent_accumulated[1] + self.local_offset[1] - base_offset[1]
        )
        
        # Special handling for _rw layers: calculate position through parent chain
        if self.gameobject.m_Name.endswith('_rw') and hasattr(self, 'rect_transform'):
            if root is None:
                root = self
                while root.parent:
                    root = root.parent
            
            base_name = self.gameobject.m_Name[:-3]  # Remove '_rw' suffix
            
            # Check if root itself is the base layer
            if root.gameobject.m_Name == base_name and root.image:
                base_layer = root
            else:
                base_layer = root.findChildLayer(base_name)
            
            # If base layer found, calculate proper offset through parent chain
            if base_layer and hasattr(base_layer, 'rect_transform'):
                # Start from base layer's position + center offset (anchor point for its children)
                # Base layer might not be at (0,0), use its global_offset
                base_x = base_layer.global_offset[0] if hasattr(base_layer, 'global_offset') else 0.0
                base_y = base_layer.global_offset[1] if hasattr(base_layer, 'global_offset') else 0.0
                current_x = base_x + base_layer.size[0] / 2
                current_y = base_y + base_layer.size[1] / 2
                
                log.debug(f"_rw '{self.gameobject.m_Name}': base_layer '{base_layer.gameobject.m_Name}' at global_offset={base_layer.global_offset if hasattr(base_layer, 'global_offset') else 'N/A'}, starting from ({current_x}, {current_y})")
                
                # Walk up from _rw layer to base layer, collecting intermediate containers
                containers = []
                node = self.parent
                while node and node != base_layer:
                    if hasattr(node, 'rect_transform') and node.rect_transform:
                        containers.append(node)
                    node = node.parent
                
                # Apply container offsets in order (from base toward _rw)
                for container in reversed(containers):
                    # Container's anchored_position is offset from parent center
                    # In Unity coords: +X=right, +Y=up
                    # In screen coords: +X=right, +Y=down (flip Y)
                    anchor_pos = container.rect_transform.anchored_position
                    current_x += anchor_pos[0]
                    current_y -= anchor_pos[1]  # flip Y
                
                # Now apply _rw layer's own anchored_position
                rw_anchor = self.rect_transform.anchored_position
                pivot_screen_x = current_x + rw_anchor[0]
                pivot_screen_y = current_y - rw_anchor[1]  # flip Y
                
                # Convert pivot position to top-left position
                # Unity uses size_delta for positioning, not the mesh-reconstructed image size
                # The pivot is defined relative to size_delta
                pivot = self.rect_transform.pivot
                size_for_pivot = self.size  # Use size_delta (stored in self.size)
                
                # Calculate top-left of the RectTransform logical bounds
                rect_top_x = pivot_screen_x - pivot[0] * size_for_pivot[0]
                rect_top_y = pivot_screen_y - (1 - pivot[1]) * size_for_pivot[1]
                
                # If mesh is larger than size_delta, it extends upward (in Unity Y-up)
                # which is negative Y in screen coords. Subtract the overflow.
                if self.image:
                    overflow_y = max(0, self.image.size[1] - size_for_pivot[1])
                    self.global_offset = (rect_top_x, rect_top_y - overflow_y)
                else:
                    self.global_offset = (rect_top_x, rect_top_y)
                
                log.debug(f"Calculated _rw layer '{self.gameobject.m_Name}' offset via parent chain: {self.global_offset}")
        
        # For full-canvas background layers (_bj), align Y to 0 to match root
        # This fixes cases where Unity data has small Y offsets that don't match ground truth
        elif (self.gameobject.m_Name.endswith('_bj') and 
            self.size == root_size and
            abs(self.global_offset[1]) < 20):  # Only for small offsets
            self.global_offset = (self.global_offset[0], 0.0)
        
        # Calculate accumulated offset to pass to children
        # Children's positions are relative to this node's local_offset
        new_accumulated = (
            parent_accumulated[0] + self.local_offset[0],
            parent_accumulated[1] + self.local_offset[1]
        )
        
        for child in self.children:
            child.calculateGlobalOffset(base_offset, new_accumulated, root_size)


    def getBounds(self) -> tuple[float, float, float, float]:
        """Get bounding box of this layer and all children.
        
        Returns (min_x, min_y, max_x, max_y) that encompasses all content.
        Layers may have negative offsets extending beyond (0,0) origin.
        """
        min_x, min_y = (0.0, 0.0)
        max_x, max_y = (0.0, 0.0)
        
        if self.image:
            # Use actual image size for bounds (mesh may be larger than size_delta)
            img_width = self.image.size[0]
            img_height = self.image.size[1]
            
            min_x = min(min_x, self.global_offset[0])
            min_y = min(min_y, self.global_offset[1])
            max_x = max(max_x, self.global_offset[0] + img_width)
            max_y = max(max_y, self.global_offset[1] + img_height)
        
        for child in self.children:
            c_min_x, c_min_y, c_max_x, c_max_y = child.getBounds()
            min_x = min(min_x, c_min_x)
            min_y = min(min_y, c_min_y)
            max_x = max(max_x, c_max_x)
            max_y = max(max_y, c_max_y)
        
        return min_x, min_y, max_x, max_y
    
    def getBiggestSize(self) -> tuple[float, float]:
        """Get canvas size needed to fit this layer and all children.
        
        Returns (width, height) accounting for negative offsets.
        """
        min_x, min_y, max_x, max_y = self.getBounds()
        return max_x - min_x, max_y - min_y

    def printHierarchy(self, indent: int = 0):
        """Debug: print the layer hierarchy."""
        prefix = "  " * indent
        has_img = "IMG" if self.image else "---"
        extra = ""
        if hasattr(self, 'rect_transform'):
            rt = self.rect_transform
            z_pos = getattr(rt, 'z_position', 0.0)
            extra = f" anchor=({rt.anchor_min}, {rt.anchor_max}) anchorpos={rt.anchored_position} pivot={rt.pivot} z={z_pos}"
        log.debug(f"{prefix}{has_img} {self.gameobject.m_Name} size={self.size} local_off={self.local_offset} global_off={self.global_offset}{extra}")
        for child in self.children:
            child.printHierarchy(indent + 1)

    def yieldLayers(self):
        """Yield all layers with images in Unity's rendering order.
        
        Uses Z-position as primary sort (lower Z = behind), then sibling index.
        Lower Z values render first (behind), higher Z values render last (on top).
        """
        if self.image:
            yield self
        # Sort by z-position (if available), then sibling index
        def get_sort_key(child):
            z_pos = 0.0
            if hasattr(child, 'rect_transform') and hasattr(child.rect_transform, 'z_position'):
                z_pos = child.rect_transform.z_position
            return (z_pos, child.sibling_index)
        
        sorted_children = sorted(self.children, key=get_sort_key)
        for child in sorted_children:
            yield from child.yieldLayers()

    def applyScaledImages(self, scaled_images: dict[str, Image.Image]):
        """Apply scaled images from batch results."""
        if self._pending_scale_id and self._pending_scale_id in scaled_images:
            self.image = scaled_images[self._pending_scale_id]
            self._pending_scale_id = None
        for child in self.children:
            child.applyScaledImages(scaled_images)
