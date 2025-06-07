/**
 * Dashboard JavaScript - Sistema de Monitoramento
 */

class Dashboard {
    constructor() {
        this.socket = null;
        this.chart = null;
        this.isConnected = false;
        
        this.init();
    }
    
    async init() {
        console.log('🚀 Iniciando Dashboard...');
        
        // Conecta WebSocket
        this.connectWebSocket();
        
        // Carrega dados iniciais
        await this.loadInitialData();
        
        // Configura gráfico
        this.setupChart();
        
        // Atualiza dados periodicamente
        setInterval(() => this.updateData(), 30000); // 30s
    }
    
    connectWebSocket() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('✅ WebSocket conectado');
                this.isConnected = true;
                this.updateConnectionStatus();
            });
            
            this.socket.on('disconnect', () => {
                console.log('❌ WebSocket desconectado');
                this.isConnected = false;
                this.updateConnectionStatus();
            });
            
            this.socket.on('status_update', (data) => {
                console.log('📡 Dados recebidos via WebSocket:', data);
                this.updateCurrentStatus(data);
                this.updateChart();
                this.updateHistoryTable();
            });
            
        } catch (error) {
            console.error('Erro ao conectar WebSocket:', error);
        }
    }
    
    updateConnectionStatus() {
        const statusElement = document.getElementById('connectionStatus');
        
        if (this.isConnected) {
            statusElement.textContent = '🟢 Conectado';
            statusElement.className = 'connected';
        } else {
            statusElement.textContent = '🔴 Desconectado';
            statusElement.className = 'disconnected';
        }
    }
    
    async loadInitialData() {
        try {
            // Carrega status atual
            const currentResponse = await fetch('/api/current-status');
            const currentData = await currentResponse.json();
            
            if (currentData.success && currentData.data) {
                this.updateCurrentStatus(currentData.data);
            }
            
            // Carrega histórico para tabela
            await this.updateHistoryTable();
            
        } catch (error) {
            console.error('Erro ao carregar dados iniciais:', error);
        }
    }
    
    updateCurrentStatus(data) {
        // Atualiza painel informativo
        document.getElementById('currentDay').textContent = data.day || '---';
        document.getElementById('currentClassification').textContent = data.classification || '---';
        document.getElementById('currentPeopleRisk').textContent = data.people_at_risk || '---';
        document.getElementById('currentFlooding').textContent = data.flooding || '---';
        document.getElementById('currentRainLevel').textContent = `${data.rain_level || '---'} mm`;
        
        // ✨ NOVA FUNCIONALIDADE: Aplica classes diretamente nos items individuais
        const infoItems = document.querySelectorAll('.info-item');
        
        // Remove todas as classes anteriores de todos os items
        infoItems.forEach(item => {
            item.classList.remove('item-normal', 'item-atencao', 'item-perigo');
        });
        
        // Determina qual classe aplicar baseada na classificação
        let classToAdd = 'item-normal'; // Padrão
        
        switch (data.classification?.toLowerCase()) {
            case 'atenção':
                classToAdd = 'item-atencao';
                break;
            case 'perigo':
                classToAdd = 'item-perigo';
                break;
            default:
                classToAdd = 'item-normal';
        }
        
        // Aplica a classe para TODOS os info-items
        infoItems.forEach(item => {
            item.classList.add(classToAdd);
        });
        
        // Atualiza mapa
        this.updateMap(data.classification);
        
        // Atualiza imagem da CNN
        this.updateInferenceImage(data.image_used);
        
        // Atualiza status de conexão
        this.updateConnectionStatus();
    }
    
    updateMap(classification) {
        const mapImage = document.getElementById('mapImage');
        const mapStatus = document.getElementById('mapStatus');
        
        let imagePath = '/static/images/maps/normal.jpeg';
        let statusClass = 'status-normal';
        
        switch (classification?.toLowerCase()) {
            case 'atenção':
                imagePath = '/static/images/maps/atencao.jpeg';
                statusClass = 'status-atencao';
                break;
            case 'perigo':
                imagePath = '/static/images/maps/perigo.jpeg';
                statusClass = 'status-perigo';
                break;
            default:
                imagePath = '/static/images/maps/normal.jpeg';
                statusClass = 'status-normal';
        }
        
        mapImage.src = imagePath;
        mapStatus.textContent = classification || 'Normal';
        mapStatus.className = `status-badge ${statusClass}`;
    }
    
    updateInferenceImage(imageName) {
        const inferenceImage = document.getElementById('inferenceImage');
        const imageNameElement = document.getElementById('imageName');
        
        if (imageName && imageName !== 'nenhuma') {
            inferenceImage.src = `/images/inference/${imageName}`;
            imageNameElement.textContent = imageName;
        } else {
            inferenceImage.src = '/static/images/placeholder.jpeg';
            imageNameElement.textContent = 'nenhuma';
        }
    }
    
    setupChart() {
        const ctx = document.getElementById('rainChart').getContext('2d');
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Nível de Chuva (mm)',
                    data: [],
                    borderColor: '#0066CC',
                    backgroundColor: 'rgba(0, 102, 204, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Limite de Alagamento',
                    data: [],
                    borderColor: '#FF4444',
                    backgroundColor: 'rgba(255, 68, 68, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Nível (mm)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Dia'
                        }
                    }
                }
            }
        });
        
        // Carrega dados do gráfico
        this.updateChart();
    }
    
    async updateChart() {
        try {
            const response = await fetch('/api/history');
            const historyData = await response.json();
            
            if (historyData.success && historyData.data) {
                const data = historyData.data.slice(-15);
                const labels = data.map(item => `Dia ${item.day}`);
                const rainLevels = data.map(item => item.rain_level);
                const floodingLimit = new Array(data.length).fill(50);
                
                this.chart.data.labels = labels;
                this.chart.data.datasets[0].data = rainLevels;
                this.chart.data.datasets[1].data = floodingLimit;
                this.chart.update();
            }
        } catch (error) {
            console.error('Erro ao atualizar gráfico:', error);
        }
    }
    
    async updateHistoryTable() {
        try {
            const response = await fetch('/api/recent-records');
            const historyData = await response.json();
            
            if (historyData.success && historyData.data) {
                const tbody = document.getElementById('historyTableBody');
                tbody.innerHTML = '';
                
                historyData.data.reverse().forEach(record => {
                    const row = document.createElement('tr');
                    
                    // Formata classificação com cor
                    let classificationClass = '';
                    switch (record.classification?.toLowerCase()) {
                        case 'perigo': classificationClass = 'style="color: #FF4444; font-weight: bold;"'; break;
                        case 'atenção': classificationClass = 'style="color: #FFA500; font-weight: bold;"'; break;
                        case 'normal': classificationClass = 'style="color: #78BF43; font-weight: bold;"'; break;
                    }
                    
                    row.innerHTML = `
                        <td>${record.day || '--'}</td>
                        <td>${record.rain_level || '--'}</td>
                        <td>${record.flooding || '--'}</td>
                        <td>${record.people_at_risk || '--'}</td>
                        <td ${classificationClass}>${record.classification || '--'}</td>
                        <td>${record.image_used || 'nenhuma'}</td>
                    `;
                    
                    tbody.appendChild(row);
                });
            }
        } catch (error) {
            console.error('Erro ao atualizar tabela:', error);
        }
    }
    
    async updateData() {
        if (!this.isConnected) {
            console.log('🔄 Atualizando dados via polling...');
            await this.loadInitialData();
            await this.updateChart();
        }
    }
}

// Inicializa dashboard quando página carrega
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
