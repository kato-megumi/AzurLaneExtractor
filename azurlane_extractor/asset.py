"""Unity asset bundle handling."""
import logging
from pathlib import Path
from typing import Iterable

import UnityPy
from UnityPy.classes import GameObject
from UnityPy.enums import ClassIDType

from azurlane_extractor.name_map import Skin

from .config import get_config

log = logging.getLogger(__name__)


class AzurlaneAsset:
    """Wrapper for Unity asset bundles."""
    
    def __init__(self, type: str, skin: Skin, is_censored: bool = False):
        config = get_config()
        self.directory = config.asset_dir
        self.bundle = UnityPy.load(str(config.asset_dir / Path(type) / skin.painting))
        self.container = next(iter(self.bundle.container.values()))
        self._loaded_textures = False
        if type == "painting":
            self.bundle.load([str(config.asset_dir/ type / res) for res in skin.res_list])
                

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
