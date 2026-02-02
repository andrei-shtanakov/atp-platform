# ruff: noqa: E501
"""Test site for ATP agent testing.

A simple e-commerce site with products catalog for testing web search agents.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 9876

Endpoints:
    GET /              - Home page
    GET /catalog       - Product catalog (HTML)
    GET /product/{id}  - Product detail page
    GET /about         - About page
    GET /contact       - Contact page
    GET /search?q=     - Search products
    GET /api/products  - Products API (JSON)
"""

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

app = FastAPI(title="Test Shop", description="ATP Test Site")

# Sample products data
PRODUCTS = [
    {
        "id": 1,
        "name": "Budget Laptop",
        "category": "laptop",
        "price": 499.99,
        "description": "Affordable laptop for everyday tasks",
        "specs": {"ram": "8GB", "storage": "256GB SSD", "display": "14 inch"},
    },
    {
        "id": 2,
        "name": "Pro Laptop",
        "category": "laptop",
        "price": 1299.99,
        "description": "Professional laptop for developers",
        "specs": {"ram": "16GB", "storage": "512GB SSD", "display": "15.6 inch"},
    },
    {
        "id": 3,
        "name": "Gaming Laptop",
        "category": "laptop",
        "price": 899.00,
        "description": "Gaming laptop with dedicated GPU",
        "specs": {
            "ram": "16GB",
            "storage": "512GB SSD",
            "display": "15.6 inch",
            "gpu": "RTX 3060",
        },
    },
    {
        "id": 4,
        "name": "Ultrabook Laptop",
        "category": "laptop",
        "price": 749.50,
        "description": "Thin and light ultrabook",
        "specs": {"ram": "8GB", "storage": "256GB SSD", "display": "13.3 inch"},
    },
    {
        "id": 5,
        "name": "Workstation Laptop",
        "category": "laptop",
        "price": 1899.99,
        "description": "High-performance workstation",
        "specs": {"ram": "32GB", "storage": "1TB SSD", "display": "17 inch"},
    },
    {
        "id": 6,
        "name": "Wireless Mouse",
        "category": "accessory",
        "price": 29.99,
        "description": "Ergonomic wireless mouse",
        "specs": {"connectivity": "Bluetooth", "battery": "AA"},
    },
    {
        "id": 7,
        "name": "Mechanical Keyboard",
        "category": "accessory",
        "price": 89.99,
        "description": "RGB mechanical keyboard",
        "specs": {"switches": "Cherry MX Blue", "backlight": "RGB"},
    },
    {
        "id": 8,
        "name": "USB-C Hub",
        "category": "accessory",
        "price": 49.99,
        "description": "7-in-1 USB-C hub",
        "specs": {"ports": "HDMI, USB-A x3, SD, microSD, USB-C"},
    },
    {
        "id": 9,
        "name": "Monitor 27 inch",
        "category": "display",
        "price": 349.99,
        "description": "27 inch 4K monitor",
        "specs": {"resolution": "3840x2160", "panel": "IPS", "refresh": "60Hz"},
    },
    {
        "id": 10,
        "name": "Laptop Stand",
        "category": "accessory",
        "price": 39.99,
        "description": "Adjustable aluminum laptop stand",
        "specs": {"material": "Aluminum", "adjustable": "Yes"},
    },
]

COMPANY_INFO = {
    "name": "Test Shop Inc.",
    "founded": 2020,
    "email": "support@testshop.local",
    "phone": "+1-555-123-4567",
    "address": "123 Test Street, Demo City, TC 12345",
}


def base_template(title: str, content: str) -> str:
    """Wrap content in base HTML template."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Test Shop</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ background: #333; color: white; padding: 20px; margin: -20px -20px 20px; }}
        header a {{ color: white; margin-right: 20px; text-decoration: none; }}
        header a:hover {{ text-decoration: underline; }}
        .product-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px;
            display: inline-block;
            width: 250px;
            vertical-align: top;
        }}
        .product-name {{ font-size: 18px; font-weight: bold; margin: 0 0 10px; }}
        .price {{ color: #e63946; font-size: 20px; font-weight: bold; }}
        .category {{ color: #666; font-size: 12px; text-transform: uppercase; }}
        .search-box {{ margin: 20px 0; }}
        .search-box input {{ padding: 10px; width: 300px; font-size: 16px; }}
        .search-box button {{ padding: 10px 20px; font-size: 16px; cursor: pointer; }}
        .product-detail {{ background: white; padding: 20px; border-radius: 8px; }}
        .specs {{ background: #f9f9f9; padding: 15px; border-radius: 4px; margin-top: 15px; }}
        .specs dt {{ font-weight: bold; }}
        .specs dd {{ margin: 0 0 10px 0; }}
    </style>
</head>
<body>
    <header>
        <a href="/">Home</a>
        <a href="/catalog">Catalog</a>
        <a href="/about">About</a>
        <a href="/contact">Contact</a>
    </header>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Home page."""
    content = """
    <h1>Welcome to Test Shop</h1>
    <p>Your one-stop shop for laptops and accessories.</p>
    <div class="search-box">
        <form action="/search" method="get">
            <input type="text" name="q" placeholder="Search products...">
            <button type="submit">Search</button>
        </form>
    </div>
    <h2>Featured Categories</h2>
    <ul>
        <li><a href="/catalog?category=laptop">Laptops</a></li>
        <li><a href="/catalog?category=accessory">Accessories</a></li>
        <li><a href="/catalog?category=display">Displays</a></li>
    </ul>
    """
    return base_template("Home", content)


@app.get("/catalog", response_class=HTMLResponse)
def catalog(category: str | None = None, max_price: float | None = None) -> str:
    """Product catalog page."""
    products = PRODUCTS

    # Filter by category
    if category:
        products = [p for p in products if p["category"] == category]

    # Filter by max price
    if max_price is not None:
        products = [p for p in products if p["price"] <= max_price]

    items_html = ""
    for p in products:
        items_html += f'''
        <div class="product-card" data-category="{p["category"]}" data-price="{p["price"]}">
            <span class="category">{p["category"]}</span>
            <h3 class="product-name">{p["name"]}</h3>
            <p>{p["description"]}</p>
            <span class="price">${p["price"]}</span>
            <br><br>
            <a href="/product/{p["id"]}">View Details</a>
        </div>
        '''

    filter_info = ""
    if category:
        filter_info += f" in category '{category}'"
    if max_price:
        filter_info += f" under ${max_price}"

    content = f"""
    <h1>Product Catalog</h1>
    <p>Showing {len(products)} products{filter_info}</p>
    <div class="search-box">
        <form action="/search" method="get">
            <input type="text" name="q" placeholder="Search products...">
            <button type="submit">Search</button>
        </form>
    </div>
    <div class="products">
        {items_html}
    </div>
    """
    return base_template("Catalog", content)


@app.get("/product/{product_id}", response_class=HTMLResponse)
def product_detail(product_id: int) -> str:
    """Product detail page."""
    product = next((p for p in PRODUCTS if p["id"] == product_id), None)

    if not product:
        return base_template("Not Found", "<h1>Product Not Found</h1>")

    specs_html = ""
    for key, value in product["specs"].items():
        specs_html += f"<dt>{key}</dt><dd>{value}</dd>"

    content = f"""
    <div class="product-detail">
        <span class="category">{product["category"]}</span>
        <h1 class="product-name">{product["name"]}</h1>
        <p>{product["description"]}</p>
        <span class="price">${product["price"]}</span>

        <div class="specs">
            <h3>Specifications</h3>
            <dl>
                {specs_html}
            </dl>
        </div>

        <p><a href="/catalog">Back to Catalog</a></p>
    </div>
    """
    return base_template(product["name"], content)


@app.get("/search", response_class=HTMLResponse)
def search(q: str = Query(default="")) -> str:
    """Search products."""
    query = q.lower()
    results = [
        p
        for p in PRODUCTS
        if query in p["name"].lower() or query in p["description"].lower()
    ]

    items_html = ""
    for p in results:
        items_html += f'''
        <div class="product-card" data-category="{p["category"]}">
            <span class="category">{p["category"]}</span>
            <h3 class="product-name">{p["name"]}</h3>
            <span class="price">${p["price"]}</span>
            <br><br>
            <a href="/product/{p["id"]}">View Details</a>
        </div>
        '''

    content = f"""
    <h1>Search Results</h1>
    <p>Found {len(results)} products for "{q}"</p>
    <div class="search-box">
        <form action="/search" method="get">
            <input type="text" name="q" value="{q}" placeholder="Search products...">
            <button type="submit">Search</button>
        </form>
    </div>
    <div class="products">
        {items_html if items_html else "<p>No products found.</p>"}
    </div>
    """
    return base_template("Search", content)


@app.get("/about", response_class=HTMLResponse)
def about() -> str:
    """About page."""
    content = f"""
    <h1>About Us</h1>
    <p><strong>{COMPANY_INFO["name"]}</strong> was founded in <strong>{COMPANY_INFO["founded"]}</strong>.</p>
    <p>We are dedicated to providing quality tech products at competitive prices.</p>
    <h2>Our Mission</h2>
    <p>To make technology accessible to everyone.</p>
    """
    return base_template("About", content)


@app.get("/contact", response_class=HTMLResponse)
def contact() -> str:
    """Contact page."""
    content = f"""
    <h1>Contact Us</h1>
    <p><strong>Email:</strong> <a href="mailto:{COMPANY_INFO["email"]}">{COMPANY_INFO["email"]}</a></p>
    <p><strong>Phone:</strong> {COMPANY_INFO["phone"]}</p>
    <p><strong>Address:</strong> {COMPANY_INFO["address"]}</p>
    """
    return base_template("Contact", content)


# API endpoints for programmatic access
@app.get("/api/products")
def api_products(
    category: str | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
) -> list[dict]:
    """Get products as JSON."""
    products = PRODUCTS

    if category:
        products = [p for p in products if p["category"] == category]
    if max_price is not None:
        products = [p for p in products if p["price"] <= max_price]
    if min_price is not None:
        products = [p for p in products if p["price"] >= min_price]

    return products


@app.get("/api/products/{product_id}")
def api_product(product_id: int) -> dict | None:
    """Get single product as JSON."""
    return next((p for p in PRODUCTS if p["id"] == product_id), None)


@app.get("/api/company")
def api_company() -> dict:
    """Get company info as JSON."""
    return COMPANY_INFO


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "products_count": len(PRODUCTS)}
