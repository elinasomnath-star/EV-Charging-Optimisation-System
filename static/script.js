document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const corridorNameEl = document.getElementById('corridor-name');
    const currentStepEl = document.getElementById('current-step');
    const totalStepsEl = document.getElementById('total-steps');
    const currentHourEl = document.getElementById('current-hour');
    const sideCurrentStepEl = document.getElementById('side-current-step');
    const sideTotalStepsEl = document.getElementById('side-total-steps');
    const sideCurrentHourEl = document.getElementById('side-current-hour');
    const eventLogEl = document.getElementById('event-log');
    const stationsContainer = document.getElementById('stations-container');
    const modelStateEl = document.getElementById('model-state');
    const runBadgeEl = document.getElementById('run-badge');
    const modelCard = document.querySelector('.model-card');

    const kpiServed = document.getElementById('kpi-served');
    const kpiMissed = document.getElementById('kpi-missed');
    const kpiCost = document.getElementById('kpi-cost');
    const kpiReward = document.getElementById('kpi-reward');

    const metricsTableBody = document.getElementById('metrics-table-body');

    let eventSource = null;
    let stationsMap = new Map();

    startBtn.addEventListener('click', () => {
        if (eventSource) {
            eventSource.close();
        }

        resetDashboardForRun();
        eventSource = new EventSource('/api/simulation/stream');

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
                addLogEntry(`Model error: ${data.error}`, true);
                setRunState('ERROR', 'Fault');
                eventSource.close();
                resetButton();
                return;
            }

            if (data.type === 'init') {
                handleInit(data);
            } else if (data.type === 'step') {
                handleStep(data);
            } else if (data.type === 'finish') {
                handleFinish(data);
                eventSource.close();
                resetButton();
            }
        };

        eventSource.onerror = () => {
            addLogEntry('Connection to the simulation stream was interrupted.', true);
            setRunState('ERROR', 'Fault');
            eventSource.close();
            resetButton();
        };
    });

    function resetDashboardForRun() {
        eventLogEl.innerHTML = '';
        stationsContainer.innerHTML = '<div class="loading-state">Initializing station configuration</div>';
        metricsTableBody.innerHTML = '<tr><td colspan="6">Model run in progress...</td></tr>';
        stationsMap.clear();

        startBtn.disabled = true;
        startBtn.textContent = 'Running';
        setRunState('RUNNING', 'Streaming');

        updateStepDisplays(0, 96, '0.0h');
        kpiServed.textContent = '0';
        kpiMissed.textContent = '0';
        kpiCost.textContent = '0';
        kpiReward.textContent = '0.0';
    }

    function resetButton() {
        startBtn.disabled = false;
        startBtn.textContent = 'Run Again';
    }

    function setRunState(state, badgeText) {
        modelStateEl.textContent = state;
        runBadgeEl.textContent = badgeText;
        modelCard.classList.toggle('is-running', state === 'RUNNING');
    }

    function updateStepDisplays(step, totalSteps, hourText) {
        currentStepEl.textContent = step;
        sideCurrentStepEl.textContent = step;
        totalStepsEl.textContent = `/ ${totalSteps}`;
        sideTotalStepsEl.textContent = `/ ${totalSteps}`;
        currentHourEl.textContent = hourText;
        sideCurrentHourEl.textContent = hourText;
    }

    function addLogEntry(message, isError = false) {
        const li = document.createElement('li');
        li.className = `log-entry${isError ? ' error' : ''}`;
        li.textContent = message;
        eventLogEl.appendChild(li);

        if (eventLogEl.children.length > 50) {
            eventLogEl.removeChild(eventLogEl.firstChild);
        }
        eventLogEl.scrollTop = eventLogEl.scrollHeight;
    }

    function handleInit(data) {
        corridorNameEl.textContent = data.corridor_name;
        updateStepDisplays(0, data.total_steps, '0.0h');

        stationsContainer.innerHTML = '';
        data.stations.forEach((station) => {
            stationsMap.set(station.idx, station);
            stationsContainer.appendChild(createStationNode(station));
        });

        addLogEntry(`Initialized ${data.corridor_name} corridor with ${data.stations.length} charging stations.`);
    }

    function createStationNode(station) {
        const node = document.createElement('div');
        node.className = 'station-node';
        node.id = `station-${station.idx}`;

        node.innerHTML = `
            <div class="station-info">
                <div class="station-name">${formatStationName(station.name)}</div>
                <div class="station-meta">
                    <span>${station.location_km} km</span>
                    <span>${station.n_chargers} chargers</span>
                    <span>${station.charger_kw} kW</span>
                    <span>${station.charger_type}</span>
                </div>
            </div>
            <div class="queue-container">
                <span class="queue-label">Queue</span>
                <div class="queue-bar-bg">
                    <div class="queue-bar-fill" id="queue-fill-${station.idx}"></div>
                </div>
                <span class="queue-count" id="queue-count-${station.idx}">0</span>
            </div>
        `;

        return node;
    }

    function handleStep(data) {
        updateStepDisplays(data.step, totalStepsEl.textContent.replace('/ ', ''), `${data.hour}h`);

        if (data.new_evs > 0) {
            addLogEntry(`Step ${data.step}: ${data.new_evs} EVs routed to ${data.routes}.`);
        }

        data.queues.forEach((qSize, idx) => {
            const fillEl = document.getElementById(`queue-fill-${idx}`);
            const countEl = document.getElementById(`queue-count-${idx}`);

            if (!fillEl || !countEl) {
                return;
            }

            countEl.textContent = qSize;
            const percentage = Math.min((qSize / 20) * 100, 100);
            fillEl.style.width = `${percentage}%`;
            fillEl.className = 'queue-bar-fill';

            if (qSize >= 15) {
                fillEl.classList.add('high');
            } else if (qSize >= 7) {
                fillEl.classList.add('medium');
            }
        });

        const totals = data.metrics.reduce((acc, metric) => {
            acc.served += metric.served;
            acc.missed += metric.missed;
            acc.cost += metric.cost;
            acc.reward += metric.reward;
            return acc;
        }, { served: 0, missed: 0, cost: 0, reward: 0 });

        updateKpis(totals.served, totals.missed, totals.cost, totals.reward);
    }

    function handleFinish(data) {
        addLogEntry(`Simulation complete: ${data.total_routed} EVs routed through the corridor.`);
        updateKpis(data.total_served, data.total_missed, data.total_cost, data.total_reward);
        setRunState('COMPLETE', 'Complete');

        metricsTableBody.innerHTML = '';
        data.station_metrics.forEach((metric, idx) => {
            const stationName = stationsMap.get(idx)?.name || `Station ${idx + 1}`;
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${formatStationName(stationName)}</td>
                <td>${metric.served}</td>
                <td>${metric.missed}</td>
                <td>Rs ${Math.round(metric.cost).toLocaleString()}</td>
                <td>${metric.reward.toFixed(2)}</td>
                <td>${metric.violations}</td>
            `;
            metricsTableBody.appendChild(tr);
        });
    }

    function updateKpis(served, missed, cost, reward) {
        kpiServed.textContent = served;
        kpiMissed.textContent = missed;
        kpiCost.textContent = Math.round(cost).toLocaleString();
        kpiReward.textContent = reward.toFixed(2);
    }

    function formatStationName(name) {
        return String(name).replaceAll('_', ' ');
    }
});
