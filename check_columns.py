"""Debug script to check the columns of the 'pet' table in the database."""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

uri = os.environ.get("SUPABASE_DB_URI")

conn = psycopg2.connect(uri)
with conn.cursor() as cur:
    cur.execute(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'pet'",
    )
    for _row in cur.fetchall():
        pass
conn.close()
