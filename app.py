
import os
import sys
import tempfile
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import torch
    import ultralytics
except Exception as e:
    print("YOLO libraries not available (%s); classical method only." % e)

from flask import Flask, request, send_file, render_template_string
import flow2drawio
from flow2drawio import convert

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlowCraft | Editable Diagrams</title>

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        :root {
            --bg: #0b1020;
            --panel: #121a2d;
            --panel-light: #18233b;
            --border: #27334d;
            --text: #edf2ff;
            --muted: #94a3b8;
            --accent: #8b5cf6;
            --accent-light: #a78bfa;
            --cyan: #22d3ee;
            --green: #34d399;
        }

        body {
            min-height: 100vh;
            background:
                radial-gradient(ellipse at 15% 5%, #272052 0, transparent 38%),
                radial-gradient(ellipse at 90% 85%, #102c43 0, transparent 35%),
                var(--bg);
            color: var(--text);
            font-family: Inter, "Segoe UI", Arial, sans-serif;
            line-height: 1.6;
            padding: 28px 18px 45px;
        }

        .container {
            width: 100%;
            max-width: 1050px;
            margin: 0 auto;
        }

        .navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 15px;
            margin-bottom: 65px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 19px;
            font-weight: 750;
            letter-spacing: -0.5px;
        }

        .brand-icon {
            width: 43px;
            height: 43px;
            display: grid;
            place-items: center;
            border-radius: 14px;
            background: linear-gradient(135deg, #8b5cf6, #4f46e5);
            box-shadow: 0 7px 25px #8b5cf633;
            font-size: 22px;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 13px;
            border: 1px solid #263b3c;
            border-radius: 30px;
            background: #112522;
            color: #86efac;
            font-size: 12px;
            font-weight: 600;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            background: #34d399;
            border-radius: 50%;
            box-shadow: 0 0 10px #34d39988;
        }

        .hero {
            text-align: center;
            margin-bottom: 38px;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #c4b5fd;
            background: #8b5cf618;
            border: 1px solid #8b5cf633;
            border-radius: 30px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 20px;
        }

        h1 {
            font-size: clamp(32px, 5vw, 53px);
            line-height: 1.12;
            letter-spacing: -2px;
            font-weight: 800;
            margin-bottom: 19px;
        }

        .gradient-text {
            background: linear-gradient(100deg, #a78bfa, #67e8f9);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .hero p {
            max-width: 610px;
            margin: 0 auto;
            color: var(--muted);
            font-size: 15px;
        }

        .workspace {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 0.9fr);
            gap: 22px;
            align-items: stretch;
        }

        .card {
            background: linear-gradient(145deg, #151e32, #101727);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 27px;
            box-shadow: 0 20px 65px #00000020;
        }

        .card-heading {
            display: flex;
            align-items: center;
            gap: 13px;
            margin-bottom: 24px;
        }

        .heading-icon {
            width: 43px;
            height: 43px;
            display: grid;
            place-items: center;
            flex-shrink: 0;
            border-radius: 13px;
            background: #8b5cf620;
            color: #c4b5fd;
            font-size: 21px;
        }

        .card-heading h2 {
            font-size: 17px;
            font-weight: 700;
        }

        .card-heading p {
            color: var(--muted);
            font-size: 12px;
            margin-top: 2px;
        }

        .upload-zone {
            position: relative;
            min-height: 245px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 25px 16px;
            border: 1.5px dashed #465477;
            border-radius: 17px;
            background: #0d1425;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s,
                        transform 0.2s;
        }

        .upload-zone:hover,
        .upload-zone.dragover {
            border-color: var(--accent-light);
            background: #17152d;
            transform: translateY(-2px);
        }

        .upload-icon {
            width: 64px;
            height: 64px;
            display: grid;
            place-items: center;
            background: linear-gradient(145deg, #8b5cf626, #22d3ee12);
            border: 1px solid #8b5cf644;
            border-radius: 19px;
            color: #c4b5fd;
            font-size: 28px;
            margin-bottom: 17px;
        }

        .upload-zone h3 {
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .upload-zone p {
            color: var(--muted);
            font-size: 12px;
        }

        .browse {
            color: #a78bfa;
            font-weight: 700;
        }

        .file-input {
            position: absolute;
            width: 1px;
            height: 1px;
            opacity: 0;
            pointer-events: none;
        }

        .file-info {
            display: none;
            align-items: center;
            gap: 12px;
            margin-top: 14px;
            padding: 12px;
            border: 1px solid #30415b;
            background: #172238;
            border-radius: 12px;
            overflow: hidden;
        }

        .file-symbol {
            width: 38px;
            height: 38px;
            flex-shrink: 0;
            display: grid;
            place-items: center;
            border-radius: 10px;
            background: #22d3ee15;
            color: var(--cyan);
            font-size: 19px;
        }

        .file-details {
            min-width: 0;
            flex: 1;
        }

        .file-name {
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            font-size: 12px;
            font-weight: 700;
        }

        .file-size {
            color: var(--muted);
            font-size: 11px;
        }

        .remove-file {
            border: none;
            background: transparent;
            color: #fca5a5;
            cursor: pointer;
            font-size: 19px;
            padding: 5px;
        }

        .section-label {
            font-size: 12px;
            color: #cbd5e1;
            font-weight: 700;
            margin: 25px 0 12px;
        }

        .mode-list {
            display: grid;
            gap: 10px;
        }

        .mode-option {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px;
            border: 1px solid #2b3852;
            border-radius: 14px;
            background: #10182a;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
        }

        .mode-option:hover {
            border-color: #6552a8;
        }

        .mode-option.selected {
            border-color: #8b5cf6;
            background: #8b5cf612;
            box-shadow: inset 0 0 0 1px #8b5cf620;
        }

        .mode-option input {
            accent-color: #8b5cf6;
            width: 16px;
            height: 16px;
            flex-shrink: 0;
        }

        .mode-icon {
            width: 39px;
            height: 39px;
            flex-shrink: 0;
            display: grid;
            place-items: center;
            border-radius: 11px;
            background: #ffffff08;
            font-size: 19px;
        }

        .mode-copy {
            flex: 1;
            min-width: 0;
        }

        .mode-title {
            font-size: 13px;
            font-weight: 700;
        }

        .mode-description {
            color: var(--muted);
            font-size: 11px;
            margin-top: 2px;
        }

        .mode-badge {
            align-self: flex-start;
            padding: 4px 7px;
            border-radius: 6px;
            background: #8b5cf620;
            color: #c4b5fd;
            font-size: 9px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }

        .convert-button {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            width: 100%;
            padding: 15px;
            margin-top: 23px;
            border: none;
            border-radius: 13px;
            background: linear-gradient(100deg, #7c3aed, #6366f1);
            box-shadow: 0 8px 25px #7c3aed30;
            color: white;
            font-family: inherit;
            font-size: 14px;
            font-weight: 750;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s;
        }

        .convert-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px #7c3aed50;
        }

        .convert-button:disabled {
            opacity: 0.65;
            cursor: wait;
            transform: none;
        }

        .result-card {
            display: flex;
            flex-direction: column;
        }

        .preview-area {
            flex: 1;
            min-height: 255px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 20px;
            border-radius: 17px;
            border: 1px solid #293650;
            background-color: #0e1525;
            background-image: radial-gradient(#34415b 1px, transparent 1px);
            background-size: 19px 19px;
        }

        .diagram-illustration {
            width: 100%;
            max-width: 285px;
            margin-bottom: 24px;
        }

        .preview-area h3 {
            font-size: 15px;
            margin-bottom: 7px;
        }

        .preview-area p {
            max-width: 280px;
            color: var(--muted);
            font-size: 12px;
        }

        .features {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 20px;
        }

        .feature {
            padding: 14px 7px;
            text-align: center;
            background: #ffffff04;
            border: 1px solid #27334a;
            border-radius: 13px;
        }

        .feature-icon {
            display: block;
            font-size: 20px;
            margin-bottom: 7px;
        }

        .feature strong {
            display: block;
            font-size: 11px;
        }

        .feature span:last-child {
            display: block;
            color: var(--muted);
            font-size: 10px;
            margin-top: 3px;
        }

        .error-message {
            margin-top: 18px;
            padding: 13px 15px;
            border: 1px solid #7f303d;
            border-radius: 12px;
            background: #451c2b55;
            color: #fecaca;
            font-size: 13px;
            overflow-wrap: anywhere;
        }

        .error-message strong {
            display: block;
            margin-bottom: 3px;
        }

        .footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 35px;
            padding-top: 22px;
            border-top: 1px solid #202b40;
            color: #77859d;
            font-size: 11px;
        }

        .footer span {
            display: flex;
            align-items: center;
            gap: 7px;
        }

        .footer a {
            color: #a5b4fc;
            text-decoration: none;
        }

        .footer a:hover {
            color: #c4b5fd;
        }

        @media (max-width: 760px) {
            body {
                padding: 20px 14px 30px;
            }

            .navbar {
                margin-bottom: 43px;
            }

            .workspace {
                grid-template-columns: 1fr;
            }

            .hero {
                margin-bottom: 28px;
            }

            .card {
                padding: 21px;
            }

            .preview-area {
                min-height: 280px;
            }
        }

        @media (max-width: 390px) {
            .status {
                font-size: 10px;
                padding: 7px 9px;
            }

            .brand {
                font-size: 16px;
            }

            .mode-badge {
                display: none;
            }

            .card {
                padding: 16px;
            }
        }
    </style>
</head>

<body>
    <div class="container">

        <nav class="navbar">
            <div class="brand">
                <div class="brand-icon">⌘</div>
                <span>Flow<span class="gradient-text">Craft</span></span>
            </div>

            <div class="status">
                <span class="status-dot"></span>
                Local Processing
            </div>
        </nav>

        <header class="hero">
            <div class="eyebrow">
                <span>✦</span>
                IMAGE TO DIAGRAM
            </div>

            <h1>
                Turn flowcharts into<br>
                <span class="gradient-text">editable diagrams.</span>
            </h1>

            <p>
                Upload your flowchart image and transform it into an
                editable draw.io diagram. Choose your detection method
                and let the conversion begin.
            </p>
        </header>

        <main class="workspace">

            <section class="card">
                <div class="card-heading">
                    <div class="heading-icon">↥</div>
                    <div>
                        <h2>Upload your image</h2>
                        <p>Start with a flowchart or diagram image</p>
                    </div>
                </div>

                <form id="convertForm" method="post"
                      enctype="multipart/form-data">

                    <label class="upload-zone" id="uploadZone"
                           for="imageInput">
                        <div class="upload-icon">⇧</div>
                        <h3>Drop your image here</h3>
                        <p>
                            or <span class="browse">browse files</span>
                        </p>
                        <p style="margin-top: 12px">
                            PNG, JPG, JPEG, WEBP
                        </p>
                    </label>

                    <input class="file-input" type="file"
                           id="imageInput" name="img"
                           accept="image/*" required>

                    <div class="file-info" id="fileInfo">
                        <div class="file-symbol">▧</div>
                        <div class="file-details">
                            <div class="file-name" id="fileName"></div>
                            <div class="file-size" id="fileSize"></div>
                        </div>
                        <button type="button" class="remove-file"
                                id="removeFile"
                                aria-label="Remove selected image">×</button>
                    </div>

                    <div class="section-label">DETECTION METHOD</div>

                    <div class="mode-list">

                        <label class="mode-option selected">
                            <input type="radio" name="method"
                                   value="auto" checked>
                            <div class="mode-icon">⚡</div>
                            <div class="mode-copy">
                                <div class="mode-title">Auto Detect</div>
                                <div class="mode-description">
                                    YOLO with classical fallback
                                </div>
                            </div>
                            <span class="mode-badge">RECOMMENDED</span>
                        </label>

                        <label class="mode-option">
                            <input type="radio" name="method"
                                   value="dl">
                            <div class="mode-icon">◈</div>
                            <div class="mode-copy">
                                <div class="mode-title">YOLO</div>
                                <div class="mode-description">
                                    Deep learning detection only
                                </div>
                            </div>
                        </label>

                        <label class="mode-option">
                            <input type="radio" name="method"
                                   value="classical">
                            <div class="mode-icon">⌗</div>
                            <div class="mode-copy">
                                <div class="mode-title">Classical</div>
                                <div class="mode-description">
                                    Traditional computer vision methods
                                </div>
                            </div>
                        </label>

                    </div>

                    <button class="convert-button" type="submit"
                            id="convertButton">
                        <span>✧</span>
                        <span id="buttonText">Convert to Diagram</span>
                        <span>→</span>
                    </button>

                </form>

                {% if msg %}
                <div class="error-message" role="alert">
                    <strong>Conversion notice</strong>
                    {{ msg }}
                </div>
                {% endif %}
            </section>

            <section class="card result-card">
                <div class="card-heading">
                    <div class="heading-icon"
                         style="background:#22d3ee18;color:#67e8f9">
                        ◫
                    </div>
                    <div>
                        <h2>What you'll get</h2>
                        <p>A diagram you can open and edit</p>
                    </div>
                </div>

                <div class="preview-area">

                    <svg class="diagram-illustration"
                         viewBox="0 0 320 210"
                         xmlns="http://www.w3.org/2000/svg"
                         role="img"
                         aria-label="Illustration of a connected flowchart">

                        <defs>
                            <linearGradient id="purple"
                                x1="0" y1="0" x2="1" y2="1">
                                <stop offset="0%" stop-color="#a78bfa"/>
                                <stop offset="100%" stop-color="#7c3aed"/>
                            </linearGradient>

                            <linearGradient id="cyan"
                                x1="0" y1="0" x2="1" y2="1">
                                <stop offset="0%" stop-color="#67e8f9"/>
                                <stop offset="100%" stop-color="#0891b2"/>
                            </linearGradient>

                            <marker id="arrow" markerWidth="7"
                                markerHeight="7" refX="5" refY="3"
                                orient="auto">
                                <path d="M0,0 L6,3 L0,6 Z"
                                      fill="#8291b4"/>
                            </marker>
                        </defs>

                        <g fill="none" stroke="#8291b4"
                           stroke-width="2"
                           marker-end="url(#arrow)">
                            <path d="M160 46 V65"/>
                            <path d="M160 98 V115 H82 V130"/>
                            <path d="M160 115 H238 V130"/>
                            <path d="M82 163 V178 H160 V190"/>
                            <path d="M238 163 V178 H160"/>
                        </g>

                        <rect x="111" y="12" width="98" height="34"
                              rx="10" fill="#192641"
                              stroke="url(#purple)" stroke-width="1.5"/>

                        <text x="160" y="33" fill="#e9ddff"
                              font-size="11" text-anchor="middle"
                              font-family="Arial">Start</text>

                        <path d="M132 65 H188 L202 81.5 L188 98 H132
                                 L118 81.5 Z"
                              fill="#1c2541" stroke="url(#purple)"
                              stroke-width="1.5"/>

                        <text x="160" y="85" fill="#ddd6fe"
                              font-size="10" text-anchor="middle"
                              font-family="Arial">Process</text>

                        <rect x="37" y="130" width="90" height="33"
                              rx="9" fill="#102e3c"
                              stroke="url(#cyan)" stroke-width="1.5"/>

                        <text x="82" y="150" fill="#a5f3fc"
                              font-size="10" text-anchor="middle"
                              font-family="Arial">Option A</text>

                        <rect x="193" y="130" width="90" height="33"
                              rx="9" fill="#252044"
                              stroke="#a78bfa" stroke-width="1.5"/>

                        <text x="238" y="150" fill="#ddd6fe"
                              font-size="10" text-anchor="middle"
                              font-family="Arial">Option B</text>

                        <rect x="116" y="190" width="88" height="16"
                              rx="7" fill="#172d36"
                              stroke="#34d399" stroke-width="1.2"/>

                    </svg>

                    <h3>From image to structure</h3>
                    <p>
                        Convert your diagram into a draw.io file,
                        then move shapes, edit text, and adjust
                        connections in the diagram editor.
                    </p>
                </div>

                <div class="features">
                    <div class="feature">
                        <span class="feature-icon">◈</span>
                        <strong>Editable</strong>
                        <span>Separate elements</span>
                    </div>

                    <div class="feature">
                        <span class="feature-icon">⇄</span>
                        <strong>Connected</strong>
                        <span>Diagram edges</span>
                    </div>

                    <div class="feature">
                        <span class="feature-icon">↓</span>
                        <strong>Exportable</strong>
                        <span>Draw.io format</span>
                    </div>
                </div>
            </section>

        </main>

        <footer class="footer">
            <span>✦ FlowCraft · Image to Editable Diagram</span>
            <span>
                Opens with
                <a href="https://app.diagrams.net/"
                   target="_blank" rel="noopener noreferrer">
                    draw.io ↗
                </a>
            </span>
        </footer>

    </div>

    <script>
        const input = document.getElementById("imageInput");
        const zone = document.getElementById("uploadZone");
        const fileInfo = document.getElementById("fileInfo");
        const fileName = document.getElementById("fileName");
        const fileSize = document.getElementById("fileSize");
        const removeFile = document.getElementById("removeFile");
        const form = document.getElementById("convertForm");
        const convertButton = document.getElementById("convertButton");
        const buttonText = document.getElementById("buttonText");

        function showFile(file) {
            if (!file) return;

            if (!file.type.startsWith("image/")) {
                alert("Please select a valid image file.");
                input.value = "";
                fileInfo.style.display = "none";
                return;
            }

            fileName.textContent = file.name;

            const size = file.size < 1024 * 1024
                ? (file.size / 1024).toFixed(1) + " KB"
                : (file.size / (1024 * 1024)).toFixed(2) + " MB";

            fileSize.textContent = size;
            fileInfo.style.display = "flex";
            zone.querySelector("h3").textContent = "Image selected";
            zone.querySelector("p").innerHTML =
                '<span class="browse">Choose another image</span>';
        }

        input.addEventListener("change", function () {
            showFile(this.files[0]);
        });

        ["dragenter", "dragover"].forEach(function (eventName) {
            zone.addEventListener(eventName, function (event) {
                event.preventDefault();
                zone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(function (eventName) {
            zone.addEventListener(eventName, function (event) {
                event.preventDefault();
                zone.classList.remove("dragover");
            });
        });

        zone.addEventListener("drop", function (event) {
            const files = event.dataTransfer.files;

            if (files.length) {
                const transfer = new DataTransfer();
                transfer.items.add(files[0]);
                input.files = transfer.files;
                showFile(files[0]);
            }
        });

        removeFile.addEventListener("click", function () {
            input.value = "";
            fileInfo.style.display = "none";
            zone.querySelector("h3").textContent =
                "Drop your image here";
            zone.querySelector("p").innerHTML =
                'or <span class="browse">browse files</span>';
        });

        document.querySelectorAll(".mode-option").forEach(function (option) {
            option.addEventListener("change", function () {
                document.querySelectorAll(".mode-option").forEach(
                    function (item) {
                        item.classList.toggle(
                            "selected",
                            item.querySelector("input").checked
                        );
                    }
                );
            });
        });

        form.addEventListener("submit", function (event) {
            if (!input.files.length) {
                event.preventDefault();
                alert("Please upload an image first.");
                return;
            }

            convertButton.disabled = true;
            buttonText.textContent = "Converting...";
        });
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files["img"]
        path = os.path.join(tempfile.mkdtemp(), "in.png")
        f.save(path)

        try:
            xml, n, m = convert(path, request.form.get("method", "auto"))
        except Exception as e:
            traceback.print_exc()
            return render_template_string(PAGE, msg=f"Failed: {e}")

        if n == 0:
            return render_template_string(
                PAGE,
                msg="No boxes found in this image."
            )

        print(
            f"converted: {n} nodes, {m} edges "
            f"(detector: {convert.last['method']})"
        )

        out = os.path.join(os.path.dirname(path), "diagram.drawio")

        with open(out, "w", encoding="utf-8") as output_file:
            output_file.write(xml)

        return send_file(
            out,
            as_attachment=True,
            download_name="diagram.drawio"
        )

    return render_template_string(PAGE, msg=None)


if __name__ == "__main__":
    demo = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "flowchart2.png"
    )

    if os.path.exists(demo):
        try:
            convert(demo, "auto")
            print("warm-up OK (detector: %s)" % convert.last["method"])
        except Exception:
            traceback.print_exc()
            print(
                "warm-up failed - the page will still start, "
                "but use 'Classical only' until this is fixed."
            )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=False
    )
