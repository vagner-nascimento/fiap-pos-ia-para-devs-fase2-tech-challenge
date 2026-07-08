@echo off
docker compose -f docker-compose.yml down
docker compose -f docker-compose.yml up --build -d
