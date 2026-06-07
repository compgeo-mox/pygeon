import xml.etree.ElementTree as ET


def get_coverage_percentage(file):
    tree = ET.parse(file)
    root = tree.getroot()
    line_rate = float(root.attrib["line-rate"])
    return line_rate * 100


baseline = 95
current = get_coverage_percentage("coverage_report.xml")

print(f"Baseline coverage: {baseline:.2f}%")
print(f"PR coverage: {current:.2f}%")

if current < baseline:
    print("❌ Coverage below the mandatory level of {:}%.".format(baseline))
    exit(1)
else:
    print("✅ Coverage above the mandatory level of {:}%.".format(baseline))
    exit(0)
