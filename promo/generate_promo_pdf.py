"""Generate FX-Range-Master promotional PDF for NotebookLM"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, Image
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os

# Colors
BG_DARK = HexColor('#0b1120')
SURFACE = HexColor('#131a2e')
CARD = HexColor('#192038')
ACCENT = HexColor('#4f8eff')
GREEN = HexColor('#22c55e')
RED = HexColor('#ef4444')
CYAN = HexColor('#06b6d4')
YELLOW = HexColor('#eab308')
TEXT = HexColor('#d1d5e4')
MUTED = HexColor('#6b7394')
WHITE = HexColor('#ffffff')
DARK_BG = HexColor('#0f1629')

output_path = os.path.join(os.path.dirname(__file__), 'FX-Range-Master-Overview.pdf')

# Custom page background
class DarkPageTemplate:
    def __init__(self):
        pass

    def draw_bg(self, canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setFillColor(BG_DARK)
        canvas_obj.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)

        # Top accent line
        canvas_obj.setStrokeColor(ACCENT)
        canvas_obj.setLineWidth(3)
        canvas_obj.line(0, letter[1] - 3, letter[0], letter[1] - 3)

        # Bottom bar
        canvas_obj.setFillColor(SURFACE)
        canvas_obj.rect(0, 0, letter[0], 30, fill=1, stroke=0)
        canvas_obj.setFillColor(MUTED)
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.drawCentredString(letter[0]/2, 12, "FX-Range-Master | AI-Powered USD/ILS Trading Intelligence | Confidential")

        canvas_obj.restoreState()

tmpl = DarkPageTemplate()

# Styles
styles = {
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=36,
                            textColor=WHITE, alignment=TA_CENTER, spaceAfter=6, leading=42),
    'subtitle': ParagraphStyle('subtitle', fontName='Helvetica', fontSize=14,
                               textColor=CYAN, alignment=TA_CENTER, spaceAfter=20, leading=18),
    'tagline': ParagraphStyle('tagline', fontName='Helvetica', fontSize=11,
                              textColor=MUTED, alignment=TA_CENTER, spaceAfter=30, leading=14),
    'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=22,
                         textColor=ACCENT, spaceAfter=12, spaceBefore=24, leading=26),
    'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=16,
                         textColor=CYAN, spaceAfter=8, spaceBefore=16, leading=20),
    'h3': ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=13,
                         textColor=GREEN, spaceAfter=6, spaceBefore=12, leading=16),
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10.5,
                           textColor=TEXT, spaceAfter=8, leading=15),
    'body_small': ParagraphStyle('body_small', fontName='Helvetica', fontSize=9,
                                  textColor=MUTED, spaceAfter=6, leading=13),
    'bullet': ParagraphStyle('bullet', fontName='Helvetica', fontSize=10.5,
                             textColor=TEXT, spaceAfter=4, leading=14,
                             leftIndent=20, bulletIndent=8),
    'metric_label': ParagraphStyle('metric_label', fontName='Helvetica', fontSize=8,
                                    textColor=MUTED, alignment=TA_CENTER, leading=10),
    'metric_value': ParagraphStyle('metric_value', fontName='Helvetica-Bold', fontSize=18,
                                    textColor=WHITE, alignment=TA_CENTER, leading=22),
    'feature_title': ParagraphStyle('feature_title', fontName='Helvetica-Bold', fontSize=11,
                                     textColor=WHITE, spaceAfter=3, leading=14),
    'feature_desc': ParagraphStyle('feature_desc', fontName='Helvetica', fontSize=9,
                                    textColor=MUTED, leading=12),
    'center': ParagraphStyle('center', fontName='Helvetica', fontSize=11,
                              textColor=TEXT, alignment=TA_CENTER, leading=14),
}

doc = SimpleDocTemplate(
    output_path,
    pagesize=letter,
    topMargin=0.8*inch,
    bottomMargin=0.7*inch,
    leftMargin=0.75*inch,
    rightMargin=0.75*inch,
)

story = []

# ═══════════════════════════════════════
# PAGE 1: COVER
# ═══════════════════════════════════════
story.append(Spacer(1, 1.8*inch))
story.append(Paragraph("FX-RANGE-MASTER", styles['title']))
story.append(Spacer(1, 8))
story.append(HRFlowable(width="40%", thickness=2, color=ACCENT, spaceAfter=12, spaceBefore=0, hAlign='CENTER'))
story.append(Paragraph("AI-Powered USD/ILS Trading Intelligence", styles['subtitle']))
story.append(Spacer(1, 20))
story.append(Paragraph("Real-time forex analytics platform combining live market data,<br/>"
                        "machine learning signals, and comprehensive risk management<br/>"
                        "tools for USD/ILS currency pair trading.", styles['center']))
story.append(Spacer(1, 40))

# Key metrics row
metric_data = [
    [Paragraph("LIVE DATA", styles['metric_label']),
     Paragraph("AI ENGINE", styles['metric_label']),
     Paragraph("SIGNALS", styles['metric_label']),
     Paragraph("MONITORING", styles['metric_label'])],
    [Paragraph("Real-Time", styles['metric_value']),
     Paragraph("RF Model", styles['metric_value']),
     Paragraph("6+", styles['metric_value']),
     Paragraph("24/7", styles['metric_value'])],
    [Paragraph("USD/ILS rates & candles", styles['body_small']),
     Paragraph("Random Forest decisions", styles['body_small']),
     Paragraph("Technical indicators", styles['body_small']),
     Paragraph("User activity tracking", styles['body_small'])]
]

metric_table = Table(metric_data, colWidths=[1.6*inch]*4, rowHeights=[14, 30, 14])
metric_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), SURFACE),
    ('BOX', (0,0), (-1,-1), 1, HexColor('#263354')),
    ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor('#1e2d52')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 6),
    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ('ROUNDEDCORNERS', [6,6,6,6]),
]))
story.append(metric_table)

story.append(Spacer(1, 30))
story.append(Paragraph("Cloud-hosted on Google Cloud Run | Firebase Authentication | Firestore Analytics", styles['body_small']))

story.append(PageBreak())

# ═══════════════════════════════════════
# PAGE 2: DASHBOARD SCREENSHOT
# ═══════════════════════════════════════
story.append(Paragraph("Live Dashboard", styles['h1']))
story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#263354'), spaceAfter=12))
story.append(Paragraph(
    "The FX-Range-Master dashboard provides a comprehensive, real-time view of USD/ILS market data, "
    "AI-powered trading signals, and technical analysis — all in a single screen without scrolling.", styles['body']))
story.append(Spacer(1, 10))

# Add dashboard screenshot
dashboard_img_path = os.path.join(os.path.dirname(__file__), 'dashboard.png')
if os.path.exists(dashboard_img_path):
    # Get image dimensions to calculate aspect ratio
    from reportlab.lib.utils import ImageReader
    img_reader = ImageReader(dashboard_img_path)
    img_w, img_h = img_reader.getSize()
    aspect = img_h / img_w
    display_w = 6.8 * inch
    display_h = display_w * aspect

    # Add border effect using a table
    dash_img = Image(dashboard_img_path, width=display_w, height=display_h)
    img_table = Table([[dash_img]], colWidths=[display_w + 8])
    img_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor('#0a0e1a')),
        ('BOX', (0,0), (-1,-1), 2, ACCENT),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 12))

# Dashboard panel labels
story.append(Paragraph("Dashboard Panels", styles['h2']))
panel_items = [
    ("Market Overview", "Live USD/ILS price, daily change, and embedded AI Engine mini-panel with TRADE/SKIP decisions"),
    ("Price Position in Window", "Visual gauge showing price position within the trading range with distance-to-boundary percentages"),
    ("Key Levels", "Support/resistance levels (S1, S2, R1, R2), baseline, and stop boundaries with status indicators"),
    ("News Sentiment", "Real-time filtered news feed with sentiment scoring and keyword highlighting"),
    ("Candles 5m", "Intraday SVG candlestick chart with upper/lower range boundary lines"),
    ("Signal History", "Chronological log of AI-generated trading signals with direction and price context"),
]
for title, desc in panel_items:
    story.append(Paragraph(f"<bullet>&bull;</bullet> <b>{title}</b> — {desc}", styles['bullet']))

story.append(PageBreak())

# ═══════════════════════════════════════
# PAGE 3: PLATFORM OVERVIEW
# ═══════════════════════════════════════
story.append(Paragraph("Platform Overview", styles['h1']))
story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#263354'), spaceAfter=12))

story.append(Paragraph(
    "FX-Range-Master is a specialized forex analytics platform designed for USD/ILS currency pair trading. "
    "It combines real-time market data feeds, AI-powered decision signals from a Random Forest machine learning model, "
    "and comprehensive technical analysis tools into a single, intuitive dashboard.", styles['body']))

story.append(Paragraph(
    "The platform serves both individual traders and organizations, providing institutional-grade analytics "
    "with a modern, dark-themed interface optimized for extended monitoring sessions.", styles['body']))

story.append(Paragraph("Architecture", styles['h2']))

arch_data = [
    [Paragraph("<b>Component</b>", ParagraphStyle('th', parent=styles['body'], textColor=ACCENT, fontSize=9)),
     Paragraph("<b>Technology</b>", ParagraphStyle('th', parent=styles['body'], textColor=ACCENT, fontSize=9)),
     Paragraph("<b>Purpose</b>", ParagraphStyle('th', parent=styles['body'], textColor=ACCENT, fontSize=9))],
    [Paragraph("Backend", styles['body_small']), Paragraph("Python / Flask", styles['body_small']), Paragraph("API server, data processing, ML model serving", styles['body_small'])],
    [Paragraph("Frontend", styles['body_small']), Paragraph("HTML5 / CSS Grid / SVG", styles['body_small']), Paragraph("Responsive dashboard with real-time updates", styles['body_small'])],
    [Paragraph("AI Engine", styles['body_small']), Paragraph("Random Forest (scikit-learn)", styles['body_small']), Paragraph("Trade/Skip decision with confidence scoring", styles['body_small'])],
    [Paragraph("Authentication", styles['body_small']), Paragraph("Firebase Auth", styles['body_small']), Paragraph("User management, login/logout, admin roles", styles['body_small'])],
    [Paragraph("Analytics", styles['body_small']), Paragraph("Cloud Firestore", styles['body_small']), Paragraph("User activity logging, usage heatmaps", styles['body_small'])],
    [Paragraph("Hosting", styles['body_small']), Paragraph("Google Cloud Run", styles['body_small']), Paragraph("Serverless container deployment, auto-scaling", styles['body_small'])],
    [Paragraph("Data Feed", styles['body_small']), Paragraph("Yahoo Finance API", styles['body_small']), Paragraph("Live USD/ILS rates, OHLC candles, news", styles['body_small'])],
]

arch_table = Table(arch_data, colWidths=[1.3*inch, 1.8*inch, 3.2*inch])
arch_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), HexColor('#1a2440')),
    ('BACKGROUND', (0,1), (-1,-1), SURFACE),
    ('BOX', (0,0), (-1,-1), 1, HexColor('#263354')),
    ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor('#1e2d52')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('RIGHTPADDING', (0,0), (-1,-1), 8),
]))
story.append(arch_table)

story.append(Spacer(1, 16))

story.append(Paragraph("Dashboard Layout", styles['h2']))
story.append(Paragraph(
    "The dashboard uses a responsive 3-column, 2-row CSS Grid layout optimized for displaying "
    "all critical trading data without scrolling:", styles['body']))

layout_items = [
    ("Market Context Bar", "Full-width top bar showing session status, volatility level, economic events, and price position context"),
    ("Market Overview (Left)", "Current USD/ILS price with daily change percentage, integrated AI Engine mini-panel showing trade/skip decision with confidence gauge"),
    ("Price Position Gauge (Center)", "SVG semicircular gauge showing where current price sits within the daily range (0-100%), with color-coded zones"),
    ("Key Levels Table (Right)", "Support/resistance levels (S1, S2, R1, R2), daily range boundaries, and pivot points"),
    ("Market News (Bottom-Left)", "Curated news feed with timestamped headlines affecting USD/ILS"),
    ("Intraday Candles (Bottom-Center)", "SVG candlestick chart showing intraday OHLC price action"),
    ("Trading Signals (Bottom-Right)", "Technical indicators including RSI, MACD, Bollinger Bands, ATR, EMA Cross, and trend direction"),
]

for title, desc in layout_items:
    story.append(Paragraph(f"<bullet>&bull;</bullet> <b>{title}</b> - {desc}", styles['bullet']))

story.append(PageBreak())

# ═══════════════════════════════════════
# PAGE 4: AI ENGINE
# ═══════════════════════════════════════
story.append(Paragraph("AI Decision Engine", styles['h1']))
story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#263354'), spaceAfter=12))

story.append(Paragraph(
    "The core intelligence of FX-Range-Master is its AI Decision Engine, powered by a Random Forest "
    "classification model. The engine analyzes multiple market features in real-time to generate "
    "actionable trade/skip signals with confidence percentages.", styles['body']))

story.append(Paragraph("Model Features", styles['h2']))

features_data = [
    [Paragraph("<b>Feature</b>", ParagraphStyle('th', parent=styles['body'], textColor=ACCENT, fontSize=9)),
     Paragraph("<b>Description</b>", ParagraphStyle('th', parent=styles['body'], textColor=ACCENT, fontSize=9)),
     Paragraph("<b>Signal</b>", ParagraphStyle('th', parent=styles['body'], textColor=ACCENT, fontSize=9))],
    [Paragraph("Gap %", styles['body_small']), Paragraph("Opening gap from previous close as percentage", styles['body_small']), Paragraph("Market sentiment indicator", styles['body_small'])],
    [Paragraph("ATR (14)", styles['body_small']), Paragraph("Average True Range over 14 periods", styles['body_small']), Paragraph("Volatility measurement", styles['body_small'])],
    [Paragraph("RSI (14)", styles['body_small']), Paragraph("Relative Strength Index over 14 periods", styles['body_small']), Paragraph("Overbought/oversold conditions", styles['body_small'])],
    [Paragraph("Volatility", styles['body_small']), Paragraph("Current market volatility classification", styles['body_small']), Paragraph("Risk environment assessment", styles['body_small'])],
    [Paragraph("Session", styles['body_small']), Paragraph("Active trading session (NY/London/Asia)", styles['body_small']), Paragraph("Liquidity context", styles['body_small'])],
    [Paragraph("Position", styles['body_small']), Paragraph("Price position within daily range (0-100%)", styles['body_small']), Paragraph("Mean reversion signal", styles['body_small'])],
]

feat_table = Table(features_data, colWidths=[1.2*inch, 2.8*inch, 2.3*inch])
feat_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), HexColor('#1a2440')),
    ('BACKGROUND', (0,1), (-1,-1), SURFACE),
    ('BOX', (0,0), (-1,-1), 1, HexColor('#263354')),
    ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor('#1e2d52')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('RIGHTPADDING', (0,0), (-1,-1), 8),
]))
story.append(feat_table)

story.append(Spacer(1, 12))

story.append(Paragraph("Decision Output", styles['h2']))
story.append(Paragraph(
    "The AI Engine outputs one of three decisions:", styles['body']))

decisions = [
    ("TRADE (Green)", "Model confidence exceeds threshold - favorable conditions for entering a position"),
    ("SKIP (Yellow)", "Uncertain conditions - model recommends waiting for clearer signals"),
    ("N/A (Gray)", "Insufficient data or market closed - no decision available"),
]
for title, desc in decisions:
    story.append(Paragraph(f"<bullet>&bull;</bullet> <b>{title}</b> - {desc}", styles['bullet']))

story.append(Spacer(1, 12))
story.append(Paragraph(
    "Each decision includes a confidence percentage (0-100%) displayed both numerically and as a "
    "semicircular gauge with color-coded zones (red/yellow/green). The mini-panel version is embedded "
    "directly in the Market Overview card for constant visibility.", styles['body']))

story.append(Paragraph("Simulation Mode", styles['h2']))
story.append(Paragraph(
    "The platform includes a Simulation mode that allows users to test the AI Engine's decision-making "
    "in real-time. When activated via the 'Simulate' button, the dashboard pauses live data and runs "
    "through simulated market scenarios step-by-step, showing how the AI Engine responds to different "
    "price movements, volatility changes, and market conditions. Users can switch back to 'Real-Time' "
    "mode at any time to resume live data.", styles['body']))

story.append(PageBreak())

# ═══════════════════════════════════════
# PAGE 5: FEATURES
# ═══════════════════════════════════════
story.append(Paragraph("Platform Features", styles['h1']))
story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#263354'), spaceAfter=16))

features = [
    ("Live Market Data", ACCENT,
     "Real-time USD/ILS exchange rates updated every 30 seconds with intraday OHLC candlestick charts. "
     "Displays current price, daily change percentage, and price position within the trading range."),

    ("AI Decision Engine", GREEN,
     "Random Forest machine learning model analyzing gap percentage, ATR, RSI, volatility, and session "
     "data to generate TRADE/SKIP signals with confidence scoring. Integrated mini-panel for constant visibility."),

    ("Technical Signals", CYAN,
     "Six technical indicators displayed in real-time: RSI (14), MACD, Bollinger Bands position, "
     "ATR volatility, EMA crossover direction, and overall trend classification."),

    ("Key Levels & Range", YELLOW,
     "Automatically calculated support (S1, S2) and resistance (R1, R2) levels with daily range "
     "boundaries. SVG gauge showing price position within the range as a percentage."),

    ("Market News Feed", ACCENT,
     "Curated, timestamped news headlines from financial sources affecting USD/ILS pair. "
     "Headlines are filtered for relevance to forex, Israeli economy, and Fed policy."),

    ("Market Context Bar", GREEN,
     "Full-width status bar showing active trading session (NY/London/Asia), current volatility level, "
     "upcoming economic events, and price position context for quick situational awareness."),

    ("Simulation Mode", CYAN,
     "Interactive simulation running on the live dashboard. Pauses real-time data and steps through "
     "market scenarios showing AI decisions, price movements, and signal changes in real-time."),

    ("Performance Analytics", YELLOW,
     "Backtesting statistics modal showing historical AI model performance, win rate, profit factor, "
     "and strategy metrics for evaluating the decision engine's accuracy."),

    ("Admin Panel", RED,
     "Administrative interface for user management with Firebase Authentication integration. "
     "Supports user creation, deletion, role assignment, and 2FA configuration."),

    ("User Activity Monitoring", ACCENT,
     "Per-user activity tracking via Cloud Firestore. Admin monitor shows login count, total events, "
     "data views, 30-day activity heatmap, and recent event log with color-coded event types."),

    ("Firebase Authentication", GREEN,
     "Secure user authentication with Firebase Auth supporting email/password login, "
     "admin role management, and optional multi-factor authentication."),

    ("Cloud Deployment", CYAN,
     "Deployed on Google Cloud Run with automatic scaling, HTTPS, and global CDN. "
     "Docker containerized for consistent builds and zero-downtime deployments."),
]

for i, (title, color, desc) in enumerate(features):
    box_data = [[
        Paragraph(f'<font color="#{color.hexval()[2:]}">{title}</font>', styles['feature_title']),
    ], [
        Paragraph(desc, styles['feature_desc']),
    ]]
    box = Table(box_data, colWidths=[6.3*inch])
    box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), SURFACE),
        ('BOX', (0,0), (-1,-1), 1, HexColor('#263354')),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (0,0), 8),
        ('BOTTOMPADDING', (-1,-1), (-1,-1), 8),
        ('TOPPADDING', (0,1), (0,1), 0),
        ('ROUNDEDCORNERS', [4,4,4,4]),
    ]))
    story.append(KeepTogether([box, Spacer(1, 6)]))

story.append(PageBreak())

# ═══════════════════════════════════════
# PAGE 6: SECURITY & ADMIN
# ═══════════════════════════════════════
story.append(Paragraph("Security & Administration", styles['h1']))
story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#263354'), spaceAfter=12))

story.append(Paragraph("Authentication Flow", styles['h2']))
story.append(Paragraph(
    "All dashboard access requires Firebase Authentication. The platform implements a secure "
    "authentication flow with ID token verification on every API request:", styles['body']))

auth_steps = [
    "User navigates to the platform URL",
    "Firebase Auth checks authentication state via onAuthStateChanged",
    "Unauthenticated users are redirected to /login page",
    "User signs in with email/password credentials",
    "Firebase issues an ID token, stored client-side",
    "Every API call includes the ID token in the Authorization header",
    "Backend verifies the token using Firebase Admin SDK",
    "Admin role is determined by matching email against admin_emails config",
]
for i, step in enumerate(auth_steps, 1):
    story.append(Paragraph(f"<bullet>{i}.</bullet> {step}", styles['bullet']))

story.append(Spacer(1, 8))

story.append(Paragraph("Admin Capabilities", styles['h2']))
admin_features = [
    "Create new user accounts with email, display name, and password",
    "Delete user accounts from Firebase Authentication",
    "View all registered users with their details",
    "Monitor individual user activity (login frequency, page views, last active)",
    "View 30-day activity heatmap per user",
    "Access recent events log with timestamps and event details",
    "Configure multi-factor authentication via Firebase Console",
]
for feat in admin_features:
    story.append(Paragraph(f"<bullet>&bull;</bullet> {feat}", styles['bullet']))

story.append(Spacer(1, 12))

story.append(Paragraph("Activity Tracking", styles['h2']))
story.append(Paragraph(
    "The platform logs user activity to Cloud Firestore for admin monitoring. Events tracked include:", styles['body']))

events_data = [
    [Paragraph("<b>Event Type</b>", ParagraphStyle('th', parent=styles['body'], textColor=ACCENT, fontSize=9)),
     Paragraph("<b>Trigger</b>", ParagraphStyle('th', parent=styles['body'], textColor=ACCENT, fontSize=9)),
     Paragraph("<b>Details Logged</b>", ParagraphStyle('th', parent=styles['body'], textColor=ACCENT, fontSize=9))],
    [Paragraph("login", styles['body_small']), Paragraph("User opens dashboard", styles['body_small']), Paragraph("'Dashboard loaded'", styles['body_small'])],
    [Paragraph("data_refresh", styles['body_small']), Paragraph("Every 5 minutes (throttled)", styles['body_small']), Paragraph("Current price at refresh time", styles['body_small'])],
    [Paragraph("simulation_start", styles['body_small']), Paragraph("User clicks Simulate", styles['body_small']), Paragraph("'Entered simulation mode'", styles['body_small'])],
    [Paragraph("view_performance", styles['body_small']), Paragraph("User opens Performance modal", styles['body_small']), Paragraph("'Opened Performance modal'", styles['body_small'])],
]

evt_table = Table(events_data, colWidths=[1.5*inch, 2.2*inch, 2.6*inch])
evt_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), HexColor('#1a2440')),
    ('BACKGROUND', (0,1), (-1,-1), SURFACE),
    ('BOX', (0,0), (-1,-1), 1, HexColor('#263354')),
    ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor('#1e2d52')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('RIGHTPADDING', (0,0), (-1,-1), 8),
]))
story.append(evt_table)

story.append(PageBreak())

# ═══════════════════════════════════════
# PAGE 7: TECHNICAL SPECS & CLOSING
# ═══════════════════════════════════════
story.append(Paragraph("Technical Specifications", styles['h1']))
story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#263354'), spaceAfter=12))

specs = [
    ("Runtime", "Python 3.11+ on Google Cloud Run (containerized)"),
    ("Framework", "Flask with Jinja2 templating"),
    ("Frontend", "Vanilla JS, CSS Grid, SVG charts (no framework dependencies)"),
    ("Authentication", "Firebase Auth (email/password) with Admin SDK verification"),
    ("Database", "Cloud Firestore (activity logging & analytics)"),
    ("ML Model", "scikit-learn Random Forest Classifier"),
    ("Data Source", "Yahoo Finance API (yfinance library)"),
    ("Update Frequency", "30-second refresh cycle for live market data"),
    ("Deployment", "Docker container via Cloud Build, deployed to Cloud Run"),
    ("Region", "me-west1 (Tel Aviv) for optimal Israel market latency"),
    ("Scaling", "Auto-scaling 0-10 instances based on traffic"),
    ("Security", "HTTPS enforced, ID token auth on all API endpoints"),
]

for label, value in specs:
    spec_data = [[
        Paragraph(f'<font color="#{ACCENT.hexval()[2:]}">{label}</font>', styles['body_small']),
        Paragraph(value, styles['body_small']),
    ]]
    spec_tbl = Table(spec_data, colWidths=[1.5*inch, 4.8*inch])
    spec_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), HexColor('#1a2440')),
        ('BACKGROUND', (1,0), (1,0), SURFACE),
        ('BOX', (0,0), (-1,-1), 0.5, HexColor('#263354')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(spec_tbl)
    story.append(Spacer(1, 2))

story.append(Spacer(1, 30))

# Closing
story.append(HRFlowable(width="60%", thickness=2, color=ACCENT, spaceAfter=16, hAlign='CENTER'))
story.append(Paragraph("FX-RANGE-MASTER", ParagraphStyle('closing_title', parent=styles['title'], fontSize=24)))
story.append(Spacer(1, 6))
story.append(Paragraph("Trade Smarter with AI-Powered Intelligence", styles['subtitle']))
story.append(Spacer(1, 16))
story.append(Paragraph("fx-range-master-403186329512.me-west1.run.app", ParagraphStyle('url', parent=styles['center'], textColor=CYAN, fontSize=10)))
story.append(Spacer(1, 30))
story.append(Paragraph("Built with Python, Firebase, and Machine Learning", styles['body_small']))

# Build
doc.build(story, onFirstPage=tmpl.draw_bg, onLaterPages=tmpl.draw_bg)
print(f"PDF created: {output_path}")
