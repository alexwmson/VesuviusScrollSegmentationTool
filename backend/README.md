This will be cleaned up / done properly later, same for frontend and algorithm.  
For now, it will simply show how to run the main 3 containers for the API.  
  
# run api container  
```bash
docker run -d \
  --name vesuvius_api \
  --network vesuvius_net \
  -p 5000:5000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /home/alexr:/home/alexr \
  vesuvius_api:latest
```
  
# run redis container  
```bash
docker pull redis:8
docker run -d \
  --name vesuvius_redis \
  --network vesuvius_net \
  redis:8
```
  
# run celery worker  
```bash
docker run -d \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /home/alexr:/home/alexr \
  --name vesuvius_celery \
  --network vesuvius_net \
  -w /app \
  -e PYTHONPATH=/app \
  vesuvius_api:latest \
  celery -A celery_app:celery_app worker --loglevel=info
```

