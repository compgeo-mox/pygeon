import xml.etree.ElementTree as ET


def get_coverage_percentage(file):
    tree = ET.parse(file)
    root = tree.getroot()
    line_rate = float(root.attrib["line-rate"])
    return line_rate * 100


baseline = get_coverage_percentage("baseline.xml")
current = get_coverage_percentage("pr.xml")

print(f"Baseline coverage: {baseline:.2f}%")
print(f"PR coverage: {current:.2f}%")

if current < baseline:
    print("❌ Coverage has decreased!")
    exit(1)
else:
    print("✅ Coverage maintained or improved.")
    exit(0)
