FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm ci --only=production && npm install -g pm2

COPY . .

RUN touch .env

RUN npm run build

RUN mkdir -p logs

RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

RUN chown -R nextjs:nodejs /app/logs
RUN chown -R nextjs:nodejs /app/.next

USER nextjs

EXPOSE 3000

CMD ["pm2-runtime", "start", "ecosystem.config.js"] 