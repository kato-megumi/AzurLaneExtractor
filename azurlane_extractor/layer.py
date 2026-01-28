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

        # Initialize size from rect_transform for layers without mesh (e.g., face layer)
        # This ensures correct offset calculation before image is loaded
        if hasattr(self, 'rect_transform'):
            self.size = self.rect_transform.size_delta
        else:
            self.size = (0, 0)

        # Try to load image from mesh-based component first
        self.meshimage = asset.getComponentFromObject(gameobject, types=[ClassIDType.MonoBehaviour], attributes={"mMesh"})
        if self.meshimage:
            self._loadImage(self.meshimage)
        else:
            # Fallback: try Unity UI Image component
            self.uiimage = asset.getComponentFromObject(gameobject, types=[ClassIDType.MonoBehaviour], attributes={"m_Sprite"})
            if self.uiimage:
                self._loadImageFromUI(self.uiimage)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.gameobject.name}>"

    def _save_texture(self, image: Image.Image):
        """Save unscaled layer image to debug output directory."""
        config = get_config()
        if config.save_textures:
            layer_output_dir = config.output_dir / "_textures"
            layer_output_dir.mkdir(parents=True, exist_ok=True)
            layer_path = layer_output_dir / f"{self.texture2d.m_Name}.png"
            image.save(layer_path)
            log.debug(f"Saved unscaled layer: {layer_path.name}")

    def _upscale_image(self, image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
        """Upscale an image using the global upscaler if available, otherwise use LANCZOS."""
        from .extractor import get_upscaler
        
        upscaler = get_upscaler()
        if upscaler:
            # Use AI upscaler
            upscaled = upscaler.upscale(image)
            # Resize to exact target size if needed
            if upscaled.size != target_size:
                upscaled = upscaled.resize(target_size, Image.LANCZOS)
            log.debug(f"AI SCALE '{self.gameobject.m_Name}': {image.size} -> {upscaled.size}")
            return upscaled
        else:
            # Fallback to LANCZOS
            log.debug(f"SCALE Layer '{self.gameobject.m_Name}': {image.size} -> {target_size} (LANCZOS)")
            return image.resize(target_size, Image.LANCZOS)

    def _loadImageFromUI(self, uiimage):
        """Load image from Unity UI Image component (without mesh)."""
        config = get_config()
        
        # Check if sprite reference is valid (m_PathID != 0)
        if not uiimage.m_Sprite or uiimage.m_Sprite.m_PathID == 0:
            return
        
        try:
            self.sprite = uiimage.m_Sprite.read()
        except FileNotFoundError as e:
            log.warning(f"  -Skipping layer '{self.gameobject.m_Name}': sprite not found ({e})")
            return
        except Exception as e:
            log.warning(f"  -Skipping layer '{self.gameobject.m_Name}': sprite read error ({e})")
            return
        
        if not self.sprite:
            return
        
        self.texture2d = self.sprite.m_RD.texture.read()
        image = self.texture2d.image
        
        log.debug(f"UI LAYER '{self.gameobject.m_Name}': sprite='{self.sprite.m_Name}', "
                  f"texture='{self.texture2d.m_Name}', size={image.size}")
        
        # Save layer before any scaling (if enabled)
        if config.save_textures:
            self._save_texture(image)
        
        # For UI Image, use rect_transform size_delta as target size
        if hasattr(self, 'rect_transform'):
            target_size = (round(self.rect_transform.size_delta[0]), round(self.rect_transform.size_delta[1]))
            if image.size != target_size and target_size[0] > 0 and target_size[1] > 0:
                if image.size[0] < target_size[0] or image.size[1] < target_size[1]:
                    # Need to upscale
                    image = self._upscale_image(image, target_size)
                else:
                    # Just resize down
                    image = image.resize(target_size, Image.LANCZOS)
        
        self.image = image
        self.size = self.image.size

    def _loadImage(self, meshimage):
        """Load and reconstruct image from mesh data."""
        config = get_config()
        
        try:
            self.sprite = meshimage.m_Sprite.read()
        except FileNotFoundError as e:
            log.warning(f"  =Skipping layer '{self.gameobject.m_Name}': sprite not found ({e})")
            return
        if not self.sprite:
            return
        
        # Replace sprite texture with censored version for texture-only censoring
        # (skins that use _tex assets instead of separate _hx mesh)
        if self.asset.is_censored and self.asset.skin.texture_only_censor:
            if self.sprite.m_Name in self.asset.skin.remap:
                sprite_name = self.asset.skin.remap[self.sprite.m_Name]
                self.sprite = self.asset.getObjectByName(sprite_name, ClassIDType.Sprite)
                self.mesh = self.asset.getObjectByName(sprite_name + "-mesh", ClassIDType.Mesh)
        
        self.texture2d = self.sprite.m_RD.texture.read()
        image = self.texture2d.image
        
        log.debug(f"LAYER '{self.gameobject.m_Name}': sprite='{self.sprite.m_Name}', "
              f"texture='{self.texture2d.m_Name}', size={image.size}")

        # Reconstruct from mesh if available
        if not hasattr(self, 'mesh'):
            self.mesh = self.asset.getObjectByPathID(meshimage.mMesh.path_id)
        if self.mesh:
            image = recon(image, self.mesh.export().splitlines())
        elif hasattr(meshimage.mMesh, 'read') and meshimage.mMesh.m_PathID != 0:
            self.mesh = meshimage.mMesh.read()
            if self.mesh:
                image = recon(image, self.mesh.export().splitlines())

        # Handle size adjustments between recon output and expected raw sprite size
        # This only applies when mesh reconstruction was done - without mesh,
        # the texture should be scaled directly to size_delta
        psizex, psizey = image.size
        pdeltax, pdeltay = self.rect_transform.size_delta
        prawx, prawy = parse(meshimage.mRawSpriteSize)

        if self.mesh and (prawx != psizex or prawy != psizey):
            # Recon output doesn't match raw sprite size
            if psizex < prawx or psizey < prawy:
                # Recon output is smaller - paste onto canvas at correct position
                paste_x, paste_y = 0, 0
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
        
        # Save layer before any scaling (if enabled)
        if config.save_textures:
            self._save_texture(image)
        
        if before_size != target_size and before_size[0] <= target_size[0] and before_size[1] <= target_size[1]:
            # Only scale UP to match size_delta, never scale DOWN
            # Uses AI upscaler if available, otherwise LANCZOS
            self.image = self._upscale_image(image, target_size)
        else:
            # Image is same size or larger - keep as-is
            self.image = image
            if before_size != target_size:
                log.debug(f"Keep mesh size '{self.gameobject.m_Name}': image={before_size}, size_delta={target_size}")
        
        # Always use size_delta for positioning calculations
        self.size = (pdeltax, pdeltay)

    def loadImageSimple(self, image: Image.Image):
        """Load a pre-processed image (for face overlays).
        
        Face images are scaled to match size_delta exactly.
        """
        pdeltax, pdeltay = self.rect_transform.size_delta
        target_w, target_h = round(pdeltax), round(pdeltay)
        
        # Scale face to match target size exactly
        if (abs(image.size[0] - target_w) / max(target_w, 1) > 0.05 or
            abs(image.size[1] - target_h) / max(target_h, 1) > 0.05):
            log.warning(f"Face size mismatch for {self.asset.skin.display_name()} : {self.asset.skin.painting}, size {image.size} target face size {(target_w,target_h)} ")
            image = image.resize((target_w, target_h), Image.LANCZOS)
        
        self.image = image
        self.size = (target_w, target_h)

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

