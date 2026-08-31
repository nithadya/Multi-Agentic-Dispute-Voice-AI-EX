# =============================================================================
# Dispute AI — Web container
#
#   1. builder — installs node deps, runs ``npm run build`` → ``ui/dist/``.
#   2. runtime — nginx-alpine serving ``dist/`` + reverse-proxying /api/*
#      to the api container (see nginx.conf).
#
# The browser only ever talks to this container; nginx handles the API
# proxy and SSE pass-through, so the SPA needs no CORS knowledge of the
# api host. Same-origin in production = simpler.
# =============================================================================

# ── Stage 1 ── build the SPA ────────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Install deps first (cached layer). package-lock.json may not exist yet
# when the user has only used npm install once — fall back to plain install.
COPY ui/package.json ui/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# Copy the rest of the UI source and build.
COPY ui/ ./
RUN npm run build


# ── Stage 2 ── nginx serving + /api proxy ───────────────────────────────────
FROM nginx:alpine AS runtime

# Drop the default site so our config is the only one.
RUN rm /etc/nginx/conf.d/default.conf
COPY docker/web/nginx.conf /etc/nginx/conf.d/default.conf

# Copy the production bundle.
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -q --spider http://localhost/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
