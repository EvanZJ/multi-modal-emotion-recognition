# Front-end (Vite + React + TypeScript + Tailwind + shadcn-like components)

This is the front-end scaffold for the project.

## Development

Run:

```bash
cd front-end
npm ci
npm run dev
```

App will be at http://localhost:5173

## Build

```bash
npm run build
```

Dist artifacts will be at `dist/`.

## Docker

Build the image and run:

```bash
docker build -t mmer-frontend:latest .

# Run
docker run --rm -p 8080:80 mmer-frontend:latest
```

Open http://localhost:8080

## Adding shadcn-ui via CLI

If you'd like to use the official shadcn UI generator, install the CLI and initialize the project:

```bash
cd front-end
npm install
npx shadcn-ui@latest init

# Example: add a component
npx shadcn-ui@latest add button
```

This will scaffold additional components and configuration that follow the official shadcn patterns (Radix + Tailwind).


