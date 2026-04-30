let chart;
let chartData = {
    labels: [],
    actuals: [],
    predictions: [],
    lowerBounds: [],
    upperBounds: []
};
let autoSimInterval = null;
let customersData = [];
let lastActualTime = null;
const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:8000' : '';

async function fetchCustomers() {
    try {
        const response = await fetch(`${API_BASE}/simulation/customers_info`);
        customersData = await response.json();

        const select = document.getElementById('customerRef');
        customersData.forEach(c => {
            const option = document.createElement('option');
            option.value = c.customer_ref;
            const suffix = c.safe_start_time ? '' : ' (insufficient data)';
            option.textContent = `Customer ${c.customer_ref}${suffix}`;
            if (!c.safe_start_time) option.disabled = true;
            select.appendChild(option);
        });
    } catch (err) {
        console.error("Failed to load customers:", err);
        updateStatus("Failed to load customers", "error");
    }
}

function onCustomerSelect() {
    const val = document.getElementById('customerRef').value;
    if (!val) return;

    const cust = customersData.find(c => c.customer_ref == val);
    if (cust && cust.safe_start_time) {
        // Use the DB-calculated safe start time (193rd row)
        const startDate = new Date(cust.safe_start_time);

        const pad = (n) => n.toString().padStart(2, '0');
        const localStr = `${startDate.getFullYear()}-${pad(startDate.getMonth() + 1)}-${pad(startDate.getDate())}T${pad(startDate.getHours())}:${pad(startDate.getMinutes())}`;

        const simTimeInput = document.getElementById('simTime');
        simTimeInput.value = localStr;

        // Reset chart - clear arrays IN-PLACE to preserve Chart.js references
        chartData.labels.length = 0;
        chartData.actuals.length = 0;
        chartData.predictions.length = 0;
        chartData.lowerBounds.length = 0;
        chartData.upperBounds.length = 0;
        chart.update();

        document.getElementById('actualKwh').textContent = '--';
        document.getElementById('predKwh').textContent = '--';
        document.getElementById('deltaKwh').textContent = '--';
        lastActualTime = null;
    }
}

function initChart() {
    const ctx = document.getElementById('predictionChart').getContext('2d');

    // Register a custom plugin to draw confidence intervals
    const confidenceIntervalPlugin = {
        id: 'confidenceInterval',
        beforeDatasetDraw(chart, args, options) {
            const { ctx, chartArea, scales: { x, y } } = chart;

            if (chartData.labels.length === 0) return;

            ctx.save();
            ctx.beginPath();

            // Move to first upper bound point
            let started = false;
            for (let i = 0; i < chartData.labels.length; i++) {
                if (chartData.upperBounds[i] !== null) {
                    const xPos = x.getPixelForValue(i); // Use index i
                    const yPos = y.getPixelForValue(chartData.upperBounds[i]);
                    if (!started) {
                        ctx.moveTo(xPos, yPos);
                        started = true;
                    } else {
                        ctx.lineTo(xPos, yPos);
                    }
                }
            }

            // Line back through lower bounds
            for (let i = chartData.labels.length - 1; i >= 0; i--) {
                if (chartData.lowerBounds[i] !== null) {
                    const xPos = x.getPixelForValue(i); // Use index i
                    const yPos = y.getPixelForValue(chartData.lowerBounds[i]);
                    ctx.lineTo(xPos, yPos);
                }
            }

            ctx.closePath();
            ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
            ctx.fill();
            ctx.restore();
        }
    };

    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = 'Inter';

    chart = new Chart(ctx, {
        type: 'line',
        plugins: [confidenceIntervalPlugin],
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    label: 'Actual (kWh)',
                    data: chartData.actuals,
                    borderColor: '#10b981',
                    backgroundColor: '#10b981',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    spanGaps: true
                },
                {
                    label: 'Predicted (kWh)',
                    data: chartData.predictions,
                    borderColor: '#3b82f6',
                    backgroundColor: '#3b82f6',
                    borderDash: [5, 5],
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: '#0f172a',
                    pointHoverRadius: 6,
                    spanGaps: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#f8fafc',
                    bodyColor: '#e2e8f0',
                    borderColor: '#334155',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(4);
                            }

                            // Add confidence interval info if this is the prediction dataset
                            if (context.datasetIndex === 1 && chartData.lowerBounds[context.dataIndex] !== null) {
                                const lower = chartData.lowerBounds[context.dataIndex].toFixed(4);
                                const upper = chartData.upperBounds[context.dataIndex].toFixed(4);
                                return [label, `95% CI: [${lower}, ${upper}]`];
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: '#334155',
                        drawBorder: false,
                        tickColor: 'transparent'
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    grid: {
                        color: '#334155',
                        drawBorder: false
                    },
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Energy (kWh)'
                    }
                }
            },
            animation: {
                duration: 400
            }
        }
    });
}

function updateStatus(message, type = 'info') {
    const badge = document.getElementById('statusBadge');
    let color = 'var(--accent-primary)';
    if (type === 'error') color = '#ef4444';
    if (type === 'success') color = 'var(--success-color)';
    if (type === 'warning') color = 'var(--warning-color)';

    badge.innerHTML = `
                <div style="width: 8px; height: 8px; border-radius: 50%; background-color: ${color};"></div>
                ${message}
            `;
}

async function simulateNextStep() {
    const customerRef = document.getElementById('customerRef').value;
    if (!customerRef) {
        alert("Please select a customer first.");
        return;
    }
    const simTimeInput = document.getElementById('simTime');
    let currentTimeStr = simTimeInput.value;

    if (!currentTimeStr) {
        alert("Please select a simulated time.");
        return;
    }

    const btn = document.getElementById('simBtn');
    btn.disabled = true;
    document.getElementById('loadingOverlay').classList.add('active');
    updateStatus('Simulating...', 'info');

    try {
        // Ensure timezone format if needed, though simple datetime string usually works
        const payload = {
            customer_ref: parseInt(customerRef),
            current_time: new Date(currentTimeStr).toISOString()
        };

        const response = await fetch(`${API_BASE}/simulation/simulate_step`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Simulation failed');
        }

        if (data.status === 'skipped') {
            updateStatus(data.reason, 'warning');
            console.warn("Simulation skipped:", data.reason);

            if (autoSimInterval) {
                // If auto-simulating, maybe we should stop or advance time?
                // For now, let's stop to let user fix the issue
                toggleAutoSimulate();
            }
            return;
        }

        // Success
        updateStatus('Prediction Updated', 'success');
        CountQueuingStrategy
        // Check if this is a duplicate response (sparse data gap)
        if (lastActualTime === data.actual_time) {
            // Data hasn't advanced — keep moving the simulated clock forward
            updateStatus('Advancing through data gap...', 'info');
            const nextDate = new Date(data.prediction_time);
            const pad = (n) => n.toString().padStart(2, '0');
            const localStr = `${nextDate.getFullYear()}-${pad(nextDate.getMonth() + 1)}-${pad(nextDate.getDate())}T${pad(nextDate.getHours())}:${pad(nextDate.getMinutes())}`;
            simTimeInput.value = localStr;
            return;
        }
        lastActualTime = data.actual_time;

        // Update stats
        document.getElementById('actualKwh').textContent = data.actual_import_kwh.toFixed(4);
        document.getElementById('predKwh').textContent = data.predicted_import_kwh.toFixed(4);

        const delta = data.actual_import_kwh - data.predicted_import_kwh;
        const deltaEl = document.getElementById('deltaKwh');
        deltaEl.textContent = Math.abs(delta).toFixed(4);
        deltaEl.style.color = delta > 0 ? '#ef4444' : '#10b981';

        // Update Chart - include date to avoid label collisions
        const fmtOpts = { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
        const actualTimeFormatted = new Date(data.actual_time).toLocaleString([], fmtOpts);
        const predTimeFormatted = new Date(data.prediction_time).toLocaleString([], fmtOpts);

        // Helper to get or create index for a label
        function getIndex(label) {
            let idx = chartData.labels.indexOf(label);
            if (idx === -1) {
                chartData.labels.push(label);
                chartData.actuals.push(null);
                chartData.predictions.push(null);
                chartData.lowerBounds.push(null);
                chartData.upperBounds.push(null);
                return chartData.labels.length - 1;
            }
            return idx;
        }

        const actualIdx = getIndex(actualTimeFormatted);
        chartData.actuals[actualIdx] = data.actual_import_kwh;

        const predIdx = getIndex(predTimeFormatted);
        chartData.predictions[predIdx] = data.predicted_import_kwh;
        chartData.lowerBounds[predIdx] = data.lower_bound;
        chartData.upperBounds[predIdx] = data.upper_bound;

        // Fill gaps for accumulated look (if value is null, use previous value)
        for (let i = 1; i < chartData.labels.length; i++) {
            if (chartData.actuals[i] === null && chartData.actuals[i - 1] !== null && i < actualIdx) {
                // Only fill past actuals
                chartData.actuals[i] = chartData.actuals[i - 1];
            }
        }

        // Keep only last 40 points for clean view
        if (chartData.labels.length > 40) {
            chartData.labels.shift();
            chartData.actuals.shift();
            chartData.predictions.shift();
            chartData.lowerBounds.shift();
            chartData.upperBounds.shift();
        }

        chart.update();

        // Advance input time to PREDICTION time (the next 15-min slot)
        const nextDate = new Date(data.prediction_time);
        const pad2 = (n) => n.toString().padStart(2, '0');
        const localStr2 = `${nextDate.getFullYear()}-${pad2(nextDate.getMonth() + 1)}-${pad2(nextDate.getDate())}T${pad2(nextDate.getHours())}:${pad2(nextDate.getMinutes())}`;
        simTimeInput.value = localStr2;

    } catch (error) {
        console.error(error);
        updateStatus(error.message, 'error');
        if (autoSimInterval) toggleAutoSimulate();
    } finally {
        btn.disabled = false;
        document.getElementById('loadingOverlay').classList.remove('active');
    }
}

function toggleAutoSimulate() {
    const autoBtn = document.getElementById('autoSimBtn');
    if (autoSimInterval) {
        clearInterval(autoSimInterval);
        autoSimInterval = null;
        autoBtn.textContent = 'Auto Run';
        autoBtn.style.backgroundColor = 'var(--border-color)';
    } else {
        simulateNextStep(); // trigger immediately
        autoSimInterval = setInterval(simulateNextStep, 3000); // every 3 seconds
        autoBtn.textContent = 'Stop Auto';
        autoBtn.style.backgroundColor = 'var(--warning-color)';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    fetchCustomers();
});