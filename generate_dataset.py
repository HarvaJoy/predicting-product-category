"""
Script to generate a realistic products dataset for product category classification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define categories and sample products
categories = {
    'Mobile Phones': [
        'Samsung Galaxy {} {}GB',
        'iPhone {} {}GB', 
        'Xiaomi Redmi {} {}GB',
        'Google Pixel {}',
        'OnePlus {} {}GB',
        'Huawei P{} Pro',
        'Motorola Moto G{}',
        'Nokia {} Dual SIM',
        'OPPO Reno{}',
        'Realme {} Pro'
    ],
    'Laptops': [
        'Dell Inspiron {} Intel Core i{}',
        'HP Pavilion {} AMD Ryzen {}',
        'Lenovo ThinkPad {} Intel i{}',
        'ASUS VivoBook {} AMD Ryzen {}',
        'MacBook {} M{} chip',
        'Acer Aspire {} Intel Core i{}',
        'MSI Gaming Laptop GF{}',
        'Surface Laptop {}',
        'LG Gram {} Ultra-Lightweight',
        'Razer Blade {} Gaming'
    ],
    'Headphones': [
        'Sony WH-{} Wireless Headphones',
        'Bose QuietComfort {} Noise Cancelling',
        'Apple AirPods {}',
        'JBL Tune {} Wireless',
        'Sennheiser HD {} Over-Ear',
        'Beats Studio {}',
        'Audio-Technica ATH-M{}',
        'Jabra Elite {}',
        'Samsung Galaxy Buds{}',
        'Anker Soundcore Life Q{}'
    ],
    'Tablets': [
        'iPad {} {}GB WiFi',
        'Samsung Galaxy Tab {} {}GB',
        'Microsoft Surface Go {}',
        'Amazon Fire HD {} Tablet',
        'Lenovo Tab M{} {}GB',
        'Huawei MatePad {} Pro',
        'ASUS ZenPad {}',
        'Xiaomi Pad {}',
        'Nokia T{} Tablet',
        'TCL TAB {} WiFi'
    ],
    'Smartwatches': [
        'Apple Watch Series {} GPS',
        'Samsung Galaxy Watch{}',
        'Fitbit Versa {}',
        'Garmin Forerunner {}',
        'Amazfit GTR {} Pro',
        'Huawei Watch GT {}',
        'Fossil Gen {} Smartwatch',
        'TicWatch Pro {}',
        'Xiaomi Mi Watch',
        'OPPO Watch {} mm'
    ],
    'Cameras': [
        'Canon EOS {} Digital Camera',
        'Nikon D{} DSLR',
        'Sony Alpha {} Mirrorless',
        'Fujifilm X-T{} Camera',
        'GoPro HERO{}',
        'Panasonic Lumix GH{}',
        'Olympus OM-D E-M{}',
        'DJI Pocket {} Gimbal Camera',
        'Canon PowerShot G{} Mark II',
        'Nikon COOLPIX P{}'
    ],
    'TVs': [
        'Samsung {} Inch 4K Smart TV',
        'LG {} Inch OLED TV',
        'Sony BRAVIA {} Inch LED',
        'TCL {} Inch QLED Android TV',
        'Hisense {} Inch 4K UHD',
        'Philips {} Inch Ambilight TV',
        'Panasonic {} Inch LED TV',
        'Xiaomi Mi TV {} Inch',
        'Toshiba {} Inch Smart TV',
        'Sharp AQUOS {} Inch'
    ],
    'Gaming Consoles': [
        'PlayStation {} Console',
        'Xbox Series {}',
        'Nintendo Switch {}',
        'Steam Deck {}GB',
        'PlayStation {} Slim',
        'Xbox One {} Edition',
        'Nintendo Switch Lite',
        'Valve Steam Deck',
        'Retro Gaming Console {}',
        'Sega Genesis Mini'
    ],
    'Speakers': [
        'JBL Flip {} Bluetooth Speaker',
        'Bose SoundLink {} Speaker',
        'Sony SRS-XB{} Wireless',
        'Ultimate Ears BOOM {}',
        'Marshall Emberton Speaker',
        'Harman Kardon Onyx {}',
        'Anker Soundcore {}',
        'Bang & Olufsen Beosound {}',
        'Sonos One (Gen {})',
        'Amazon Echo {}th Gen'
    ],
    'Computer Accessories': [
        'Logitech MX Master {} Mouse',
        'Razer BlackWidow {} Keyboard',
        'SanDisk Ultra {}GB USB',
        'Seagate {}TB External Hard Drive',
        'TP-Link Archer {} WiFi Router',
        'Belkin USB-C Hub {}',
        'Corsair K{} RGB Keyboard',
        'Dell UltraSharp {}inch Monitor',
        'HP LaserJet Pro M{} Printer',
        'Western Digital My Passport {}TB'
    ]
}

# Generate products
products_data = []
product_id = 1000

for category, templates in categories.items():
    # Generate around 300-500 products per category
    num_products = random.randint(3000, 5000)
    
    for _ in range(num_products):
        template = random.choice(templates)
        
        # Fill in template placeholders
        if '{}' in template:
            if 'GB' in template and template.count('{}') >= 2:
                # For products with capacity
                model = random.choice(['A12', 'A22', 'A32', 'A52', '11', '12', '13', '14', '15', 'X10', 'X20'])
                capacity = random.choice([32, 64, 128, 256, 512, '1TB'])
                title = template.format(model, capacity)
            elif template.count('{}') == 2:
                # For products with two parameters
                num1 = random.choice([3, 5, 7, 9, 11, 13, 15, 17])
                num2 = random.choice([3, 5, 7, 9])
                title = template.format(num1, num2)
            else:
                # For products with one parameter
                num = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
                title = template.format(num)
        else:
            title = template
        
        # Generate other fields
        merchant_id = random.randint(100, 999)
        product_code = f'PRD{product_id:06d}'
        num_views = random.randint(10, 50000)
        merchant_rating = round(random.uniform(3.0, 5.0), 1)
        
        # Generate random date from last 2 years
        days_ago = random.randint(0, 730)
        listing_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        products_data.append({
            'product_id': product_id,
            'product_title': title,
            'merchant_id': merchant_id,
            'category_label': category,
            'product_code': product_code,
            'number_of_views': num_views,
            'merchant_rating': merchant_rating,
            'listing_date': listing_date
        })
        
        product_id += 1

# Create DataFrame
df = pd.DataFrame(products_data)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
output_path = 'data/products.csv'
df.to_csv(output_path, index=False)

print(f"Dataset created successfully!")
print(f"Total products: {len(df)}")
print(f"\nCategory distribution:")
print(df['category_label'].value_counts())
print(f"\nDataset saved to: {output_path}")
print(f"\nFirst few rows:")
print(df.head(10))
