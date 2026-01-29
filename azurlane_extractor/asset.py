"""Unity asset bundle handling."""
import logging
from typing import Iterable, Optional

import UnityPy
from UnityPy.classes import GameObject
from UnityPy.enums import ClassIDType

from azurlane_extractor.name_map import Skin

from .config import get_config

log = logging.getLogger(__name__)


class AzurlaneAsset:
    """Wrapper for Unity asset bundles."""
    
    def __init__(self, skin: Skin, is_censored: bool = False):
        config = get_config()
        self.directory = config.asset_dir
        suffix = "_hx" if is_censored and not skin.texture_only_censor else ""
        painting_name = skin.painting + suffix
        self.bundle = UnityPy.load(str(config.asset_dir / "painting" / painting_name))
        self.container = next(iter(self.bundle.container.values()))
        self._loaded_textures = False
        self.is_censored = is_censored
        self.skin = skin
        full_res_list = [res for s in skin.ship.skins for res in s.res_list] if skin.ship else []
        self.bundle.load([str(config.asset_dir/ "painting" / res) for res in full_res_list])

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

    def getComponentFromObject(self, gameobject: GameObject, types: Optional[Iterable[ClassIDType]] = None,
                               names: Optional[set[str]] = None, attributes: Optional[set[str]] = None):
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
