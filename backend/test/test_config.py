from app.config import settings

print("APP_NAME:", settings.APP_NAME)
print("DEBUG:", settings.DEBUG)
print("DATABASE_URL:", settings.DATABASE_URL)
print("SECRET_KEY set:", bool(settings.SECRET_KEY))
print("MAX_WIDTH:", settings.MAX_WIDTH)

settings.ensure_dirs()
print("Storage dirs created at:", settings.STORAGE_DIR)