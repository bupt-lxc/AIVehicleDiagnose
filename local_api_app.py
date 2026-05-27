from app.config import load_settings
from api.main import create_app
app = create_app(load_settings())
