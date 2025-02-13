FROM node:16-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install

RUN mkdir node_modules/.cache && chmod -R 777 node_modules/.cache

# Start the development server
CMD ["npm", "start"]