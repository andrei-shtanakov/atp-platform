# ATP Test Site

A simple e-commerce test site for testing ATP web search agents.

## Quick Start

### Using Docker Compose (v2)

```bash
# Start the test site
docker compose up -d

# Check it's running
curl http://localhost:9876/health

# Stop the test site
docker compose down
```

### Using Podman Compose

```bash
# Start the test site
podman-compose up -d

# Check it's running
curl http://localhost:9876/health

# Stop the test site
podman-compose down
```

### Using Docker directly

```bash
# Build
docker build -t atp-test-site .

# Run
docker run -d -p 9876:9876 --name atp-test-site atp-test-site

# Stop
docker stop atp-test-site && docker rm atp-test-site
```

### Using Podman directly

```bash
# Build
podman build -t atp-test-site .

# Run
podman run -d -p 9876:9876 --name atp-test-site atp-test-site

# Stop
podman stop atp-test-site && podman rm atp-test-site
```

### Local development (without containers)

```bash
# Install dependencies
uv add fastapi uvicorn

# Run
uv run uvicorn app:app --host 0.0.0.0 --port 9876 --reload
```

## Endpoints

### HTML Pages

| Endpoint | Description |
|----------|-------------|
| `GET /` | Home page |
| `GET /catalog` | Product catalog |
| `GET /catalog?category=laptop` | Filter by category |
| `GET /catalog?max_price=1000` | Filter by max price |
| `GET /product/{id}` | Product detail page |
| `GET /search?q=gaming` | Search products |
| `GET /about` | About page (company info, founding year) |
| `GET /contact` | Contact page (email, phone, address) |

### API Endpoints (JSON)

| Endpoint | Description |
|----------|-------------|
| `GET /api/products` | All products |
| `GET /api/products?category=laptop` | Filter by category |
| `GET /api/products?max_price=1000` | Filter by price |
| `GET /api/products/{id}` | Single product |
| `GET /api/company` | Company information |
| `GET /health` | Health check |

## Sample Data

### Products (10 items)

| Name | Category | Price |
|------|----------|-------|
| Budget Laptop | laptop | $499.99 |
| Pro Laptop | laptop | $1,299.99 |
| Gaming Laptop | laptop | $899.00 |
| Ultrabook Laptop | laptop | $749.50 |
| Workstation Laptop | laptop | $1,899.99 |
| Wireless Mouse | accessory | $29.99 |
| Mechanical Keyboard | accessory | $89.99 |
| USB-C Hub | accessory | $49.99 |
| Monitor 27 inch | display | $349.99 |
| Laptop Stand | accessory | $39.99 |

### Company Info

- **Name:** Test Shop Inc.
- **Founded:** 2020
- **Email:** support@testshop.local
- **Phone:** +1-555-123-4567

## HTML Structure for Scraping

Products on catalog page have this structure:

```html
<div class="product-card" data-category="laptop" data-price="499.99">
    <span class="category">laptop</span>
    <h3 class="product-name">Budget Laptop</h3>
    <p>Affordable laptop for everyday tasks</p>
    <span class="price">$499.99</span>
    <a href="/product/1">View Details</a>
</div>
```

### CSS Selectors

| Element | Selector |
|---------|----------|
| Product card | `.product-card` |
| Product name | `.product-name` |
| Price | `.price` |
| Category | `.category` |
| Category filter | `[data-category="laptop"]` |

## Example Test Suite

```yaml
test_suite: "web_search_tests"
version: "1.0"

agents:
  - name: "search-agent"
    type: "docker"
    config:
      image: "my-search-agent:latest"

tests:
  - id: "find-laptops-under-1000"
    name: "Find laptops under $1000"
    task:
      description: |
        Go to http://localhost:9876/catalog.
        Find all laptops with price under $1000.
        Return a JSON list with name, price, and URL for each item.
        Sort results by price ascending.
      expected_artifacts: ["laptops.json"]
    constraints:
      max_steps: 20
      timeout_seconds: 60
    assertions:
      - type: "artifact_exists"
        config:
          name: "laptops.json"
```

## Integration with ATP Tests

From the project root:

```bash
# Start test site (choose one)
docker compose -f tests/fixtures/test_site/docker-compose.yml up -d
# or: podman-compose -f tests/fixtures/test_site/docker-compose.yml up -d

# Verify it's running
curl http://localhost:9876/health
# Expected: {"status":"ok","products_count":10}

# Run ATP tests
uv run atp test examples/test_suites/web_search.yaml

# Stop test site
docker compose -f tests/fixtures/test_site/docker-compose.yml down
# or: podman-compose -f tests/fixtures/test_site/docker-compose.yml down
```
