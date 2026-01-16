from celery import Celery

celery_app = Celery(
    "vesuvius",
    broker="redis://vesuvius_redis:6379/0",
    backend="redis://vesuvius_redis:6379/0",
    imports=("workers.tasks",)
)