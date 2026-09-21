from PIL import Image, ImageDraw
img = Image.new("RGB", (900, 560), "white")
d = ImageDraw.Draw(img)
lines = [
    "SHIPPING LABEL / 快递面单",
    "Order No: 20260921001",
    "Recipient: ZHANG WEI",
    "Item: Mechanical Keyboard x1",
    "Invoice total: 899.00 CNY",
    "Signature: ______________",
]
y = 40
for text in lines:
    d.text((40, y), text, fill="black")
    y += 70
d.rectangle([30, 20, 870, 540], outline="black", width=4)
d.rectangle([560, 380, 830, 470], outline="red", width=6)
d.text((610, 415), "PAID", fill="red")
img.save("/home/alex/Projects/Jev/vltest/label.png")
print("wrote label.png")
