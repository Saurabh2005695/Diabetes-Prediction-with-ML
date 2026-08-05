document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // 1. Navigation & Tabs
    // -------------------------------------------------------------
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(tc => tc.classList.remove('active'));

            btn.classList.add('active');
            const activeTab = document.getElementById(tabId);
            if (activeTab) activeTab.classList.add('active');

            if (tabId === 'analytics-tab') {
                loadAnalyticsCharts();
            }
        });
    });

    // -------------------------------------------------------------
    // 2. Synchronize Range Sliders & Numeric Inputs
    // -------------------------------------------------------------
    const features = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ];

    features.forEach(feat => {
        const numInput = document.getElementById(feat);
        const sliderInput = document.getElementById(`${feat}_slider`);

        if (numInput && sliderInput) {
            numInput.addEventListener('input', () => {
                sliderInput.value = numInput.value;
            });
            sliderInput.addEventListener('input', () => {
                numInput.value = sliderInput.value;
            });
        }
    });

    // -------------------------------------------------------------
    // 3. Preset Sample Loader
    // -------------------------------------------------------------
    let presetSamples = [];
    fetch('/api/samples')
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                presetSamples = data.samples;
            }
        })
        .catch(err => console.error("Failed to load sample presets:", err));

    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.getAttribute('data-preset'), 10);
            if (presetSamples[idx]) {
                const vals = presetSamples[idx].values;
                features.forEach(feat => {
                    if (vals[feat] !== undefined) {
                        const numInput = document.getElementById(feat);
                        const sliderInput = document.getElementById(`${feat}_slider`);
                        numInput.value = vals[feat];
                        if (sliderInput) sliderInput.value = vals[feat];
                    }
                });
                // Auto trigger prediction
                document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
            }
        });
    });

    // -------------------------------------------------------------
    // 4. Prediction Execution & Gauge Animation
    // -------------------------------------------------------------
    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const gaugeFill = document.getElementById('gaugeFill');
    const riskPercent = document.getElementById('riskPercent');
    const statusBadge = document.getElementById('statusBadge');
    const statusDesc = document.getElementById('statusDesc');
    const factorList = document.getElementById('factorList');
    const resultModelTag = document.getElementById('resultModelTag');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Build Payload
        const payload = {};
        features.forEach(feat => {
            const val = parseFloat(document.getElementById(feat).value);
            payload[feat] = isNaN(val) ? 0 : val;
        });

        predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Calculating...';
        predictBtn.disabled = true;

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            predictBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Calculate Risk Assessment';
            predictBtn.disabled = false;

            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            renderPredictionResults(data);

        } catch (err) {
            predictBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Calculate Risk Assessment';
            predictBtn.disabled = false;
            alert("Network error executing prediction: " + err);
        }
    });

    function renderPredictionResults(data) {
        const pct = data.risk_percentage;
        riskPercent.textContent = `${pct}%`;
        riskPercent.style.color = data.color_code;

        // Animate SVG gauge: Max dashoffset is 251.2
        const strokeDash = 251.2;
        const offset = strokeDash - (strokeDash * (pct / 100));
        gaugeFill.style.strokeDashoffset = offset;
        gaugeFill.style.stroke = data.color_code;

        // Status Badge & Banner
        statusBadge.textContent = `${data.risk_severity} RISK - ${data.label.toUpperCase()}`;
        statusBadge.style.backgroundColor = `${data.color_code}25`; // 15% opacity hex
        statusBadge.style.color = data.color_code;
        statusBadge.style.border = `1px solid ${data.color_code}50`;

        statusDesc.textContent = data.assessment;

        // Model Tag
        resultModelTag.textContent = `Model: ${data.model_used}`;

        // Render Factors Breakdown
        factorList.innerHTML = '';
        data.risk_contributions.forEach(item => {
            const div = document.createElement('div');
            const isElevated = item.status === 'Elevated';
            div.className = `factor-item ${isElevated ? 'elevated' : 'normal'}`;

            div.innerHTML = `
                <div>
                    <div class="factor-name">${item.label}</div>
                    <div class="factor-sub">Normal: ${item.normal_range} ${item.unit}</div>
                </div>
                <div class="factor-val" style="color: ${isElevated ? 'var(--danger)' : 'var(--success)'}">
                    ${item.value} ${item.unit}
                    ${isElevated ? `<span style="font-size: 0.7rem;">(+${item.pct_diff}%)</span>` : ''}
                </div>
            `;
            factorList.appendChild(div);
        });
    }

    // -------------------------------------------------------------
    // 5. Custom Model File Upload / Drag & Drop
    // -------------------------------------------------------------
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('modelFileInput');
    const uploadAlert = document.getElementById('uploadAlert');
    const resetModelBtn = document.getElementById('resetModelBtn');
    const activeModelText = document.getElementById('activeModelText');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        const formData = new FormData();
        formData.append('file', file);

        uploadAlert.style.display = 'block';
        uploadAlert.className = 'alert-box';
        uploadAlert.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Uploading & validating model...';

        try {
            const res = await fetch('/api/upload-model', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (data.success) {
                uploadAlert.className = 'alert-box success';
                uploadAlert.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${data.message} (${data.model_type})`;
                activeModelText.textContent = `Custom Model: ${data.filename}`;
            } else {
                uploadAlert.className = 'alert-box error';
                uploadAlert.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> ${data.error}`;
            }
        } catch (err) {
            uploadAlert.className = 'alert-box error';
            uploadAlert.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> Upload failed: ${err}`;
        }
    }

    resetModelBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/reset-model', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                activeModelText.textContent = `Model: LogisticRegression (91.6% AUC)`;
                uploadAlert.style.display = 'block';
                uploadAlert.className = 'alert-box success';
                uploadAlert.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${data.message}`;
            }
        } catch (err) {
            console.error("Reset error:", err);
        }
    });

    // -------------------------------------------------------------
    // 6. Model Analytics Charts (Chart.js)
    // -------------------------------------------------------------
    let importanceChartInstance = null;
    let comparisonChartInstance = null;

    async function loadAnalyticsCharts() {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();

            if (!data.success || !data.metrics) return;

            const meta = data.metrics;
            const featImp = meta.feature_importances || [];

            // 1. Feature Importance Chart
            const ctxImp = document.getElementById('importanceChart').getContext('2d');
            if (importanceChartInstance) importanceChartInstance.destroy();

            importanceChartInstance = new Chart(ctxImp, {
                type: 'bar',
                data: {
                    labels: featImp.map(f => f.feature),
                    datasets: [{
                        label: 'Feature Importance Weight',
                        data: featImp.map(f => f.importance),
                        backgroundColor: 'rgba(99, 102, 241, 0.7)',
                        borderColor: '#6366f1',
                        borderWidth: 1,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { ticks: { color: '#f3f4f6' }, grid: { display: false } }
                    }
                }
            });

            // 2. Classifier Comparison Chart
            const modelsData = meta.all_models || {};
            const modelNames = Object.keys(modelsData);
            const aucScores = modelNames.map(m => modelsData[m].roc_auc);
            const accuracyScores = modelNames.map(m => modelsData[m].accuracy);

            const ctxComp = document.getElementById('comparisonChart').getContext('2d');
            if (comparisonChartInstance) comparisonChartInstance.destroy();

            comparisonChartInstance = new Chart(ctxComp, {
                type: 'bar',
                data: {
                    labels: modelNames,
                    datasets: [
                        {
                            label: 'ROC-AUC',
                            data: aucScores,
                            backgroundColor: 'rgba(20, 184, 166, 0.7)',
                            borderColor: '#14b8a6',
                            borderWidth: 1,
                            borderRadius: 6
                        },
                        {
                            label: 'Accuracy',
                            data: accuracyScores,
                            backgroundColor: 'rgba(236, 72, 153, 0.7)',
                            borderColor: '#ec4899',
                            borderWidth: 1,
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#f3f4f6' } }
                    },
                    scales: {
                        x: { ticks: { color: '#f3f4f6' }, grid: { display: false } },
                        y: {
                            min: 0.5,
                            max: 1.0,
                            ticks: { color: '#9ca3af' },
                            grid: { color: 'rgba(255,255,255,0.05)' }
                        }
                    }
                }
            });

        } catch (err) {
            console.error("Failed loading analytics metrics:", err);
        }
    }
});
