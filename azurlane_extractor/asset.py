"""Unity asset bundle handling."""
import logging
from pathlib import Path
from typing import Iterable

import UnityPy
from UnityPy.classes import GameObject
from UnityPy.enums import ClassIDType

from .constants import strip_variant_suffix

log = logging.getLogger(__name__)


class AzurlaneAsset:
    """Wrapper for Unity asset bundles."""
    
    def __init__(self, offset_dir: Path, asset_sub_path: Path):
        self.directory = offset_dir
        self.asset_sub_path = asset_sub_path
        self.bundle = UnityPy.load(str(offset_dir / asset_sub_path))
        self.container = next(iter(self.bundle.container.values()))
        self._loaded_textures = False

    def loadDependencies(self):
        """Load texture dependencies for this painting."""
        if self._loaded_textures:
            return
        
        painting_name = self.asset_sub_path.stem
        painting_dir = self.directory / self.asset_sub_path.parent
        base_name, is_variant = strip_variant_suffix(painting_name)[:2]
        is_variant = bool(is_variant)
        
        # Collect texture files
        texture_files = []
        for tex_file in painting_dir.iterdir():
            tex_name = tex_file.stem
            if not tex_name.endswith("_tex"):
                continue
            tex_base = tex_name[:-4]
            if (tex_base == painting_name or tex_base.startswith(painting_name + "_") or
                tex_base == base_name or tex_base.startswith(base_name + "_")):
                texture_files.append(str(tex_file))
        
        files_to_load = list(set(texture_files))
        
        # For variants, also load the base painting bundle
        if is_variant:
            base_painting_file = painting_dir / base_name
            if base_painting_file.exists():
                files_to_load.append(str(base_painting_file))
                log.debug(f"DEP {painting_name}: also loading base painting bundle '{base_name}'")
        
        if files_to_load:
            log.debug(f"DEP {painting_name}: loading {len(texture_files)} texture files" + 
                  (" + base bundle" if is_variant else ""))
            self.bundle.load(files_to_load)
        
        self._loaded_textures = True

    def getObjectByPathID(self, pathid: int):
        for unity_object in self.bundle.objects:
            if unity_object.path_id == pathid:
                return unity_object.read()
        return None

    def getObjectByName(self, name: str, objtype: ClassIDType):
        for unity_object in self.bundle.objects:
            if unity_object.type == objtype:
                obj = unity_object.read()
                if obj.m_Name == name:
                    return obj
        return None

    def getComponentFromObject(self, gameobject: GameObject, types: Iterable[ClassIDType] = None,
                               names: set[str] = None, attributes: set[str] = None):
        for component_pptr in gameobject.m_Components:
            if types and component_pptr.type not in types:
                continue

            component = self.getObjectByPathID(component_pptr.path_id)
            if component is None:
                continue
            if names and component.name not in names:
                continue
            if hasattr(component, 'read_typetree'):
                component.read_typetree()
            
            if attributes:
                for attribute in attributes:
                    if hasattr(component, attribute):
                        return component
                continue
            return (component, component_pptr.type)
        return None
