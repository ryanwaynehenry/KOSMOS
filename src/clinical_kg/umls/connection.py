# connection.py
import mysql.connector
from ..config import DBConfig

def create_connection(cfg: DBConfig):
    return mysql.connector.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=cfg.database,
    )
