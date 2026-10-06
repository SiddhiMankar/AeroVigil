import os
import base64
from pathlib import Path
import streamlit.components.v1 as components

_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
_component_func = components.declare_component("engine_3d", path=_frontend_dir)

def get_glb_data_uri(filepath: str) -> str:
    """Reads a GLB file and returns it as a Base64 Data URI."""
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode('utf-8')
    return f"data:model/gltf-binary;base64,{b64}"

def engine_3d_component(snapshot: dict, glb_path: str = "assets/3d/aerovigil_engine.glb", key=None):
    """
    Streamlit component that renders an interactive 3D Aero-Piston Engine Digital Twin.
    """
    glb_uri = get_glb_data_uri(glb_path)
    # The component receives the snapshot and the base64-encoded GLB
    return _component_func(snapshot=snapshot, glb_uri=glb_uri, key=key)
