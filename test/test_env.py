import os

from dotenv import load_dotenv

load_dotenv()

print(os.getenv("MINERU_API_TOKEN"))
print(os.getenv("MINERU_BASE_URL"))

