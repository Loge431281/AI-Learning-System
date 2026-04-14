# app.py - 完整的网页版聊天AI（带记忆功能）

import random
import copy
import json
import os
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ============================================================
# 数字版你（用于模拟筛选）
# ============================================================

class DigitalYou:
    def __init__(self):
        self.history = []
        self.rhetorical_patterns = ['你觉得呢', '你怎么看', '你认为呢', '你说呢', '是吧']
    
    def evaluate_response(self, response, context):
        score = 50
        for pattern in self.rhetorical_patterns:
            if pattern in response:
                score -= 50
                break
        if any(word in response for word in ['是', '不是', '因为', '所以', '可以', '需要']):
            score += 15
        if '你好' in response or 'hello' in response.lower():
            score += 10
        return max(-100, min(100, score))


# ============================================================
# 小AI（回复策略）
# ============================================================

class DirectReplyAI:
    def __init__(self):
        self.name = "DirectAI"
        self.fitness = 0
        self.type = "direct"
    
    def generate(self, user_input, history):
        greetings = ['hello', 'hi', 'hey', '你好']
        if any(g in user_input.lower() for g in greetings):
            return "你好！有什么我可以帮你的吗？"
        if '?' in user_input or '？' in user_input:
            return "这是个好问题。我来帮你解答。"
        return "嗯，我明白了。"
    
    def copy(self):
        return copy.deepcopy(self)
    
    def to_dict(self):
        return {'type': 'direct', 'name': self.name, 'fitness': self.fitness}
    
    @staticmethod
    def from_dict(data):
        ai = DirectReplyAI()
        ai.name = data.get('name', 'DirectAI')
        ai.fitness = data.get('fitness', 0)
        return ai


class NormalAI:
    def __init__(self):
        self.name = "NormalAI"
        self.fitness = 0
        self.type = "normal"
    
    def generate(self, user_input, history):
        if any(g in user_input.lower() for g in ['hello', 'hi', '你好']):
            return "你好！今天怎么样？"
        return "原来如此。能再说说吗？"
    
    def copy(self):
        return copy.deepcopy(self)
    
    def to_dict(self):
        return {'type': 'normal', 'name': self.name, 'fitness': self.fitness}
    
    @staticmethod
    def from_dict(data):
        ai = NormalAI()
        ai.name = data.get('name', 'NormalAI')
        ai.fitness = data.get('fitness', 0)
        return ai


class QuestionAI:
    def __init__(self):
        self.name = "QuestionAI"
        self.fitness = 0
        self.type = "question"
    
    def generate(self, user_input, history):
        return "你觉得呢？"
    
    def copy(self):
        return copy.deepcopy(self)
    
    def to_dict(self):
        return {'type': 'question', 'name': self.name, 'fitness': self.fitness}
    
    @staticmethod
    def from_dict(data):
        ai = QuestionAI()
        ai.name = data.get('name', 'QuestionAI')
        ai.fitness = data.get('fitness', 0)
        return ai


class SimpleAI:
    def __init__(self):
        self.name = "SimpleAI"
        self.fitness = 0
        self.type = "simple"
    
    def generate(self, user_input, history):
        return "嗯。"
    
    def copy(self):
        return copy.deepcopy(self)
    
    def to_dict(self):
        return {'type': 'simple', 'name': self.name, 'fitness': self.fitness}
    
    @staticmethod
    def from_dict(data):
        ai = SimpleAI()
        ai.name = data.get('name', 'SimpleAI')
        ai.fitness = data.get('fitness', 0)
        return ai


# 小AI工厂
def create_ai_from_dict(data):
    ai_type = data.get('type')
    if ai_type == 'direct':
        return DirectReplyAI.from_dict(data)
    elif ai_type == 'normal':
        return NormalAI.from_dict(data)
    elif ai_type == 'question':
        return QuestionAI.from_dict(data)
    elif ai_type == 'simple':
        return SimpleAI.from_dict(data)
    else:
        return NormalAI()


# ============================================================
# 大AI（带记忆功能）
# ============================================================

class ChatBigAI:
    def __init__(self, population_size=4, save_file='ai_memory.json'):
        self.population_size = population_size
        self.save_file = save_file
        self.digital_you = DigitalYou()
        self.small_ais = []
        self.conversation_history = []
        self.total_interactions = 0
        
        # 尝试加载记忆
        if os.path.exists(save_file):
            self.load_memory()
        else:
            self._init_small_ais()
            print("🤖 初始化新AI种群")
    
    def _init_small_ais(self):
        """初始化小AI种群"""
        self.small_ais = []
        ai_types = [DirectReplyAI(), NormalAI(), QuestionAI(), SimpleAI()]
        for i, ai in enumerate(ai_types[:self.population_size]):
            ai.name = f"{ai.name}_{i}"
            self.small_ais.append(ai)
    
    def _simulate_and_select(self, user_input):
        """模拟筛选最佳小AI"""
        best_ai = None
        best_score = -float('inf')
        
        original_history = self.digital_you.history.copy()
        
        for ai in self.small_ais:
            self.digital_you.history = original_history.copy()
            response = ai.generate(user_input, self.conversation_history)
            feedback = self.digital_you.evaluate_response(response, user_input)
            ai.fitness = feedback
            
            if feedback > best_score:
                best_score = feedback
                best_ai = ai
        
        self.digital_you.history = original_history
        return best_ai, best_score
    
    def _evolve_small_ais(self, best_ai):
        """进化：保留最好的，变异出新的"""
        self.small_ais.sort(key=lambda ai: ai.fitness, reverse=True)
        elite_count = max(2, self.population_size // 2)
        elite = self.small_ais[:elite_count]
        
        new_population = elite.copy()
        while len(new_population) < self.population_size:
            parent = random.choice(elite)
            child = parent.copy()
            child.name = f"{child.name.split('_')[0]}_new"
            child.fitness = 0
            new_population.append(child)
        
        self.small_ais = new_population
    
    def respond(self, user_input):
        """生成回复，并更新AI状态"""
        self.total_interactions += 1
        best_ai, best_score = self._simulate_and_select(user_input)
        
        if best_ai is None:
            best_ai = self.small_ais[0]
        
        response = best_ai.generate(user_input, self.conversation_history)
        
        # 记录对话历史
        self.conversation_history.append(f"用户: {user_input}")
        self.conversation_history.append(f"AI: {response}")
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        # 进化
        self._evolve_small_ais(best_ai)
        
        # 保存记忆
        self.save_memory()
        
        return response, best_ai.name
    
    def save_memory(self):
        """保存AI状态到文件"""
        memory_data = {
            'total_interactions': self.total_interactions,
            'conversation_history': self.conversation_history[-50:],  # 只保存最近50条
            'small_ais': [ai.to_dict() for ai in self.small_ais],
            'population_size': self.population_size
        }
        try:
            with open(self.save_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            print(f"💾 AI状态已保存到 {self.save_file}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def load_memory(self):
        """从文件加载AI状态"""
        try:
            with open(self.save_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            self.total_interactions = memory_data.get('total_interactions', 0)
            self.conversation_history = memory_data.get('conversation_history', [])
            self.population_size = memory_data.get('population_size', 4)
            
            # 加载小AI种群
            self.small_ais = []
            for ai_data in memory_data.get('small_ais', []):
                ai = create_ai_from_dict(ai_data)
                self.small_ais.append(ai)
            
            # 如果加载失败，重新初始化
            if len(self.small_ais) < self.population_size:
                self._init_small_ais()
            
            print(f"📀 加载记忆成功！已进行{self.total_interactions}次对话")
            print(f"   当前种群: {[ai.name for ai in self.small_ais]}")
        except Exception as e:
            print(f"⚠️ 加载记忆失败: {e}")
            self._init_small_ais()
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'interactions': self.total_interactions,
            'current_ais': [{'name': ai.name, 'fitness': ai.fitness} for ai in self.small_ais],
            'history_length': len(self.conversation_history)
        }


# ============================================================
# 全局AI实例（带记忆）
# ============================================================

# 创建全局AI实例，会自动加载之前的记忆
chat_ai = ChatBigAI(population_size=4, save_file='ai_memory.json')


# ============================================================
# 网页界面
# ============================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>第3集聊天AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            width: 100%;
            max-width: 600px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 80vh;
            max-height: 700px;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 {
            font-size: 1.5rem;
            margin-bottom: 5px;
        }
        .chat-header p {
            font-size: 0.8rem;
            opacity: 0.9;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }
        .message.user {
            align-items: flex-end;
        }
        .message.ai {
            align-items: flex-start;
        }
        .message-bubble {
            max-width: 80%;
            padding: 10px 15px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        .user .message-bubble {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .ai .message-bubble {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .message-name {
            font-size: 0.7rem;
            color: #999;
            margin-bottom: 4px;
            margin-left: 8px;
            margin-right: 8px;
        }
        .chat-input {
            display: flex;
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
        }
        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        .chat-input input:focus {
            border-color: #667eea;
        }
        .chat-input button {
            margin-left: 10px;
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .chat-input button:hover {
            transform: scale(1.02);
        }
        .stats {
            padding: 8px 20px;
            background: #f0f0f0;
            font-size: 0.7rem;
            color: #666;
            text-align: center;
            border-top: 1px solid #e0e0e0;
        }
        .typing {
            color: #999;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 第3集聊天AI</h1>
            <p>会进化的AI | 每次对话都在成长</p>
        </div>
        <div class="chat-messages" id="messages">
            <div class="message ai">
                <span class="message-name">AI</span>
                <div class="message-bubble">你好！我是会进化的AI。每次对话我都会变得更好。你想聊点什么？</div>
            </div>
        </div>
        <div class="chat-input">
            <input type="text" id="input" placeholder="输入消息..." onkeypress="if(event.keyCode===13) sendMessage()">
            <button onclick="sendMessage()">发送</button>
        </div>
        <div class="stats" id="stats">加载中...</div>
    </div>

    <script>
        async function loadStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('stats').innerHTML = `📊 已对话 ${data.interactions} 次 | 当前策略: ${data.current_ais.map(a => a.name).join(', ')}`;
            } catch(e) {
                console.error(e);
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            if (!message) return;
            
            // 显示用户消息
            addMessage(message, 'user');
            input.value = '';
            
            // 显示输入中
            const typingDiv = addTypingIndicator();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                const data = await response.json();
                
                // 移除输入中指示器
                removeTypingIndicator(typingDiv);
                
                // 显示AI回复
                addMessage(data.response, 'ai', data.ai_name);
                
                // 更新统计
                loadStats();
            } catch(e) {
                removeTypingIndicator(typingDiv);
                addMessage('抱歉，出错了。请重试。', 'ai', 'Error');
                console.error(e);
            }
        }
        
        function addMessage(text, sender, aiName = null) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            const nameSpan = document.createElement('div');
            nameSpan.className = 'message-name';
            nameSpan.textContent = sender === 'user' ? '你' : (aiName || 'AI');
            const bubbleDiv = document.createElement('div');
            bubbleDiv.className = 'message-bubble';
            bubbleDiv.textContent = text;
            messageDiv.appendChild(nameSpan);
            messageDiv.appendChild(bubbleDiv);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return messageDiv;
        }
        
        function addTypingIndicator() {
            const messagesDiv = document.getElementById('messages');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message ai typing-message';
            typingDiv.innerHTML = '<span class="message-name">AI</span><div class="message-bubble typing">正在思考...</div>';
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return typingDiv;
        }
        
        function removeTypingIndicator(typingDiv) {
            if (typingDiv && typingDiv.remove) {
                typingDiv.remove();
            }
        }
        
        // 页面加载时获取统计
        loadStats();
        // 每30秒刷新一次统计
        setInterval(loadStats, 30000);
    </script>
</body>
</html>
'''


# ============================================================
# Flask路由
# ============================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': '消息不能为空'}), 400
    
    response, ai_name = chat_ai.respond(user_message)
    
    return jsonify({
        'response': response,
        'ai_name': ai_name
    })


@app.route('/stats', methods=['GET'])
def stats():
    stats = chat_ai.get_stats()
    return jsonify(stats)


# ============================================================
# 启动
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 第3集聊天AI - 网页版（带记忆功能）")
    print("=" * 60)
    print(f"📀 记忆文件: ai_memory.json")
    print(f"🌐 访问地址: http://localhost:5000")
    print("=" * 60)
    print("启动服务器...")
    app.run(host='0.0.0.0', port=5000, debug=False)