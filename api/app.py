"""
API Flask para o dashboard de monitoramento
"""
import os
from flask import Flask, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import json
from datetime import datetime

# Determina os caminhos corretos baseado na localização atual
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
template_folder = os.path.join(project_root, 'web', 'templates')
static_folder = os.path.join(project_root, 'web', 'static')
data_folder = os.path.join(project_root, 'data', 'cnn', 'imagens')

print(f"📁 Pasta atual: {current_dir}")
print(f"📁 Raiz do projeto: {project_root}")
print(f"📁 Templates: {template_folder}")
print(f"📁 Static: {static_folder}")
print(f"📁 Data CNN: {data_folder}")

# Verificações de existência
print(f"✅ Templates existe: {os.path.exists(template_folder)}")
print(f"✅ Static existe: {os.path.exists(static_folder)}")
print(f"✅ Data CNN existe: {os.path.exists(data_folder)}")

app = Flask(__name__, 
           template_folder=template_folder,
           static_folder=static_folder)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Gerenciador de histórico
try:
    from database.storage import HistoryManager
    history_manager = HistoryManager()
    print("✅ HistoryManager carregado com sucesso")
except ImportError as e:
    print(f"⚠️  Erro ao carregar HistoryManager: {e}")
    history_manager = None

@app.route('/')
def dashboard():
    """Página principal do dashboard"""
    return render_template('dashboard.html')

@app.route('/api/current-status')
def get_current_status():
    """Retorna status atual do sistema"""
    try:
        if history_manager is None:
            return jsonify({
                'success': False,
                'message': 'HistoryManager não disponível'
            })
            
        current_data = history_manager.get_latest_record()
        if current_data:
            return jsonify({
                'success': True,
                'data': current_data
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Nenhum dado disponível'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/history')
def get_history():
    """Retorna histórico completo para gráficos"""
    try:
        if history_manager is None:
            return jsonify({
                'success': False,
                'message': 'HistoryManager não disponível'
            })
            
        history_data = history_manager.get_all_records()
        return jsonify({
            'success': True,
            'data': history_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/recent-records')
def get_recent_records():
    """Retorna últimos 15 registros para tabela"""
    try:
        if history_manager is None:
            return jsonify({
                'success': False,
                'message': 'HistoryManager não disponível'
            })
            
        recent_data = history_manager.get_recent_records(15)
        return jsonify({
            'success': True,
            'data': recent_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/images/inference/<filename>')
def serve_inference_image(filename):
    """Serve imagens da inferência CNN"""
    try:
        return send_from_directory(data_folder, filename)
    except Exception as e:
        print(f"❌ Erro ao servir imagem {filename}: {e}")
        # Retorna imagem placeholder se não encontrar
        placeholder_path = os.path.join(static_folder, 'images', 'placeholder.jpeg')
        if os.path.exists(placeholder_path):
            return send_from_directory(os.path.join(static_folder, 'images'), 'placeholder.jpeg')
        else:
            return jsonify({'error': 'Imagem não encontrada'}), 404

@app.route('/debug/paths')
def debug_paths():
    """Rota de debug para verificar caminhos"""
    maps_folder = os.path.join(static_folder, 'images', 'maps')
    images_folder = os.path.join(static_folder, 'images')
    
    debug_info = {
        'current_dir': current_dir,
        'project_root': project_root,
        'template_folder': template_folder,
        'static_folder': static_folder,
        'data_folder': data_folder,
        'template_exists': os.path.exists(template_folder),
        'static_exists': os.path.exists(static_folder),
        'data_exists': os.path.exists(data_folder),
        'images_folder': images_folder,
        'images_exists': os.path.exists(images_folder),
        'maps_folder': maps_folder,
        'maps_exists': os.path.exists(maps_folder),
    }
    
    # Lista arquivos se as pastas existirem
    if os.path.exists(maps_folder):
        debug_info['maps_files'] = os.listdir(maps_folder)
    
    if os.path.exists(images_folder):
        debug_info['images_files'] = os.listdir(images_folder)
        
    if os.path.exists(data_folder):
        debug_info['data_files_count'] = len([f for f in os.listdir(data_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    return jsonify(debug_info)

@app.route('/test/images')
def test_images():
    """Testa se todas as imagens necessárias existem"""
    required_images = [
        'images/maps/normal.jpeg',
        'images/maps/atencao.jpeg', 
        'images/maps/perigo.jpeg',
        'images/placeholder.jpeg'
    ]
    
    results = {}
    for img in required_images:
        img_path = os.path.join(static_folder, img)
        results[img] = {
            'path': img_path,
            'exists': os.path.exists(img_path),
            'url': f'/static/{img}'
        }
    
    return jsonify({
        'static_folder': static_folder,
        'images': results
    })

@socketio.on('connect')
def handle_connect():
    """Cliente conectado ao WebSocket"""
    print('Cliente conectado ao dashboard')
    emit('connected', {'status': 'Conectado ao sistema de monitoramento'})

@socketio.on('disconnect')
def handle_disconnect():
    """Cliente desconectado do WebSocket"""
    print('Cliente desconectado do dashboard')

def broadcast_update(data):
    """Envia atualizações para todos os clientes conectados"""
    try:
        socketio.emit('status_update', data)
        print(f"📡 Dados enviados via WebSocket: Dia {data.get('day', '?')}")
    except Exception as e:
        print(f"❌ Erro ao enviar dados via WebSocket: {e}")

# Rota para servir arquivos estáticos manualmente (fallback)
@app.route('/static/<path:filename>')
def custom_static(filename):
    """Serve arquivos estáticos com debug"""
    file_path = os.path.join(static_folder, filename)
    print(f"🔍 Tentando servir: {file_path}")
    print(f"📁 Existe: {os.path.exists(file_path)}")
    
    if os.path.exists(file_path):
        return send_from_directory(static_folder, filename)
    else:
        print(f"❌ Arquivo não encontrado: {filename}")
        return jsonify({'error': f'Arquivo não encontrado: {filename}'}), 404

if __name__ == '__main__':
    print("\n🚀 Iniciando servidor Flask...")
    print(f"🌐 Dashboard: http://localhost:5000")
    print(f"🔧 Debug paths: http://localhost:5000/debug/paths")
    print(f"🖼️  Test images: http://localhost:5000/test/images")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
