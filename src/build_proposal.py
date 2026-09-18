"""Create the short, readable Milestone 1 proposal PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#4F46E5")
PALE_BLUE = colors.HexColor("#EEF2FF")
TEXT = colors.HexColor("#243447")
MUTED = colors.HexColor("#5E6B78")
PALE_GRAY = colors.HexColor("#F4F6F8")


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ProposalTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=21, leading=24, textColor=NAVY, alignment=TA_LEFT, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Kicker", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8, leading=10, textColor=BLUE, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=13, leading=15, textColor=NAVY, spaceBefore=3, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Subsection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=9.5, leading=12, textColor=NAVY, spaceBefore=4, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.8, leading=11.2, textColor=TEXT, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["BodyText"], fontName="Helvetica-Oblique",
        fontSize=7.7, leading=9.5, textColor=MUTED, alignment=TA_CENTER, spaceBefore=2,
    ))
    styles.add(ParagraphStyle(
        name="TableText", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=7.8, leading=9.5, textColor=TEXT,
    ))
    styles.add(ParagraphStyle(
        name="TableHead", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=7.8, leading=9.5, textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="Footer", parent=styles["Normal"], fontName="Helvetica",
        fontSize=7.5, leading=9, textColor=MUTED,
    ))
    return styles


def p(text: str, style):
    return Paragraph(text, style)


def table(data, widths, styles, header=True, background=PALE_GRAY):
    converted = []
    for row_index, row in enumerate(data):
        converted.append([
            value if hasattr(value, "wrap") else p(str(value), styles["TableHead" if header and row_index == 0 else "TableText"])
            for value in row
        ])
    result = Table(converted, colWidths=widths, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DEE6")),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])
        if len(converted) > 1:
            commands.append(("BACKGROUND", (0, 1), (-1, -1), background))
    result.setStyle(TableStyle(commands))
    return result


def page_header_footer(canvas, doc):
    canvas.saveState()
    width, height = letter
    canvas.setStrokeColor(colors.HexColor("#D8DEE6"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, height - 0.52 * inch, width - doc.rightMargin, height - 0.52 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.38 * inch, "KonIQ-10k | Milestone 1 proposal")
    canvas.drawRightString(width - doc.rightMargin, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build(output: str | Path = "proposal/proposal.pdf") -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / output
    sample_figure = root / "figures/representative_samples.png"
    distribution_figure = root / "figures/mos_distribution.png"
    if not sample_figure.exists() or not distribution_figure.exists():
        raise FileNotFoundError("Run the audit first so the proposal figures exist.")

    styles = make_styles()
    doc = SimpleDocTemplate(
        str(output), pagesize=letter, rightMargin=0.62 * inch, leftMargin=0.62 * inch,
        topMargin=0.72 * inch, bottomMargin=0.62 * inch,
        title="KonIQ-10k Image Quality Assessment - Project Proposal",
        author="Muhammad Ammar Sohail and teammate",
    )
    story = []

    # Page 1: problem definition and planned system.
    story.extend([
        p("MILESTONE 1 | PROJECT PROPOSAL AND EXPERIMENTAL PLAN", styles["Kicker"]),
        p("Predicting Human-Perceived Image Quality from Real-World Photos", styles["ProposalTitle"]),
        p("Team member: Muhammad Ammar Sohail (202356790)<br/>Second teammate: to be confirmed (GitHub: mkamaleldin7)", styles["BodySmall"]),
        HRFlowable(width="100%", thickness=1.1, color=BLUE, spaceBefore=2, spaceAfter=10),
        p("Project summary", styles["Section"]),
        p(
            "This project asks whether a small transfer-learning model can estimate how technically good a photograph looks to people. The model receives one photograph and predicts a single quality score. It does not need a clean reference image to compare against, so this is called blind or no-reference image quality assessment.",
            styles["BodySmall"],
        ),
        table([
            ["Task type", "Input", "Output"],
            ["Regression", "One RGB photograph", "One continuous MOS quality score"],
        ], [1.15 * inch, 2.45 * inch, 3.2 * inch], styles),
        Spacer(1, 10),
        p("Why this problem matters", styles["Section"]),
        p(
            "Photo upload and media systems often receive images that are out of focus, noisy, poorly exposed, or heavily compressed. A quality estimate could help prioritize images for review, recommend a retake, or rank uploads when no original reference image is available. The project is useful without claiming that a course model would replace human judgment.",
            styles["BodySmall"],
        ),
        p("Planned system", styles["Section"]),
        table([
            ["Step", "Simple plan"],
            ["1. Read the image", "Use the released 512x384 RGB image."],
            ["2. Predict quality", "Use an ImageNet-pretrained ResNet18 with a one-value regression head."],
            ["3. Compare with people", "Select the best checkpoint using SROCC on the validation set."],
        ], [1.35 * inch, 5.45 * inch], styles),
        Spacer(1, 10),
        p("The goal is a clean applied deep-learning experiment, not a new image-quality method. The milestone work below is limited to inspecting the data and fixing the evaluation plan before training.", styles["BodySmall"]),
    ])

    # Page 2: actual dataset inspection and sample visualization.
    story.append(PageBreak())
    story.extend([
        p("2. DATASET INSPECTION", styles["Kicker"]),
        p("KonIQ-10k dataset", styles["Section"]),
        p(
            "KonIQ-10k contains 10,073 real-world photographs selected from Flickr-related public image data. The images were rated by people for technical quality. Unlike a dataset made by adding one artificial blur to each clean image, these photographs contain natural combinations of issues such as blur, noise, exposure problems, and compression artifacts.",
            styles["BodySmall"],
        ),
        table([
            ["Dataset fact", "Observed value"],
            ["Images", "10,073 metadata-matched images (300 extra archive files ignored)"],
            ["Released image size", "512x384 RGB version used for this plan"],
            ["Target", "Released MOS score on a 0-100 scale"],
            ["Target range", "3.91 to 88.39; mean 58.73; median 62.35"],
            ["Metadata checks", "No missing metadata values or duplicate filenames"],
        ], [1.55 * inch, 5.25 * inch], styles),
        Spacer(1, 9),
        p("What the target means", styles["Subsection"]),
        p(
            "People gave ratings on a 1-5 scale. The released metadata stores the derived mean opinion score (MOS) after rescaling it to 0-100. We use the released MOS as a continuous target: a lower value means lower perceived technical quality. This is why the task is regression rather than categories such as low, medium, and high.",
            styles["BodySmall"],
        ),
        Image(str(sample_figure), width=6.2 * inch, height=4.85 * inch),
        p("Figure 1. Actual KonIQ-10k images selected from the low, middle, and high parts of the MOS range.", styles["Caption"]),
        Spacer(1, 5),
        p(
            "Access and licensing: the 512x384 archive used for the audit is listed as CC BY 4.0 on Zenodo. The underlying source photographs can have individual attribution requirements, so the repository contains code and figures only and does not redistribute the full image archive.",
            styles["BodySmall"],
        ),
    ])

    # Page 3: distribution, split, metric, and limited future experiments.
    story.append(PageBreak())
    story.extend([
        p("3. EXPERIMENTAL PLAN", styles["Kicker"]),
        p("Split, metric, and planned comparison", styles["Section"]),
        Image(str(distribution_figure), width=6.72 * inch, height=3.68 * inch),
        p("Figure 2. Distribution of the released MOS target across all 10,073 images.", styles["Caption"]),
        Spacer(1, 5),
        p("Train, validation, and test split", styles["Subsection"]),
        table([
            ["Split", "Images", "Use"],
            ["Training", "7,058", "Fit the model"],
            ["Validation", "1,000", "Choose the checkpoint and compare planned variants"],
            ["Test", "2,015", "Use once for the final report"],
        ], [1.35 * inch, 0.85 * inch, 4.6 * inch], styles),
        Spacer(1, 7),
        p("Single validation metric: Spearman rank correlation (SROCC)", styles["Subsection"]),
        p(
            "SROCC measures whether the model orders images in a similar way to human ratings. This fits the goal better than accuracy because the target is continuous and the difference between two images' exact scores is less important than whether the better-looking image is ranked higher. The test set will remain untouched until the end.",
            styles["BodySmall"],
        ),
        p("Small set of later experiments", styles["Subsection"]),
        table([
            ["Comparison", "Question"],
            ["Mean-score baseline vs. ResNet18", "Does the image model learn more than always predicting the training mean?"],
            ["Frozen backbone vs. limited fine-tuning", "Does adapting the final visual features help?"],
            ["Optional: 224x224 vs. 512x384", "Does preserving more image detail help quality prediction?"],
        ], [2.15 * inch, 4.65 * inch], styles),
        Spacer(1, 8),
        p("Limitations", styles["Subsection"]),
        p(
            "Human quality judgments are subjective, and the dataset comes from a particular online-photo collection. The project will therefore make claims about performance on KonIQ-10k rather than about every photograph or every viewer. The proposal also treats technical quality as the target, not artistic beauty.",
            styles["BodySmall"],
        ),
        p(
            "Sources: KonIQ-10k paper (Hosu et al., 2020), authors' metadata repository, and the 512x384 Zenodo data record. Supporting code and exact run instructions are included in the repository README.",
            styles["Footer"],
        ),
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=page_header_footer, onLaterPages=page_header_footer)


if __name__ == "__main__":
    build()
