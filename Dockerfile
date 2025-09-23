# Use official Julia image
FROM julia:1.10-bullseye

# Install git (and a couple of basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Inside the container, work here
WORKDIR /work

# Copy environment files first (for dependency caching)
COPY Project.toml Manifest.toml* ./

# Install all Julia dependencies
RUN julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Copy the rest of your project
COPY . .

# Default: open Julia REPL
CMD ["julia"]