FROM node:24-alpine AS build
WORKDIR /app
ARG NPM_REGISTRY=https://registry.npmjs.org
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --registry="$NPM_REGISTRY" --fetch-retries=4 --fetch-retry-mintimeout=1000 --fetch-retry-maxtimeout=10000
COPY frontend/ ./
RUN npm run build

FROM nginx:1.28-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
