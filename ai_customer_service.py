import random
import json
import re
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 枚举类
# ============================================================

class SmallAIType(Enum):
    GREEDY = "greedy"
    EXPLORE = "explore"
    BALANCED = "balanced"
    RANDOM = "random"
    MUTATION = "mutation"
    SELF_AWARE = "self_aware"
    LANGUAGE = "language"


class MemoryType(Enum):
    RULE = "rule"
    MUTATION = "mutation"
    WARNING = "warning"
    DEATH = "death"


class LanguageStyle(Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    NEUTRAL = "neutral"


# ============================================================
# 适度记忆系统（核心新增）
# ============================================================

@dataclass
class KnowledgeItem:
    """知识条目 - 带重要性评分"""
    question: str
    answer: str
    score: int = 0
    used_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    is_example: bool = False
    last_used: float = field(default_factory=lambda: datetime.now().timestamp())
    source_users: List[str] = field(default_factory=list)
    reference_count: int = 0
    last_contributor: str = ""
    
    @property
    def confidence(self) -> float:
        if self.used_count == 0:
            return 0.3 if not self.is_example else 0.2
        success_rate = self.success_count / self.used_count
        confidence = success_rate * (1 - 1 / (self.used_count + 5))
        return min(0.95, max(0.05, confidence))
    
    @property
    def importance(self) -> float:
        """计算知识的重要性（用于存储决策）"""
        imp = 0.0
        # 成功次数贡献（最高0.4）
        imp += min(0.4, self.success_count * 0.05)
        # 被引用次数贡献（最高0.3）
        imp += min(0.3, self.reference_count * 0.03)
        # 使用频率贡献（最高0.2）
        imp += min(0.2, self.used_count * 0.02)
        # 示例知识基础分
        if self.is_example:
            imp += 0.1
        # 最近使用加分
        days_since_use = (datetime.now().timestamp() - self.last_used) / 86400
        if days_since_use < 1:
            imp += 0.1
        return min(1.0, imp)
    
    def get_success_rate(self) -> float:
        if self.used_count == 0:
            return 0
        return self.success_count / self.used_count
    
    def add_contributor(self, user_id: str):
        if user_id and user_id not in self.source_users:
            self.source_users.append(user_id)
            self.reference_count = len(self.source_users)
            self.last_contributor = user_id
    
    def get_reference_text(self) -> str:
        if self.reference_count >= 5:
            return f"根据{self.reference_count}位用户的经验"
        elif self.reference_count >= 2:
            return f"根据{self.reference_count}位用户的反馈"
        elif self.reference_count == 1:
            return "有用户遇到过类似问题"
        return ""
    
    def to_dict(self) -> dict:
        return {
            'question': self.question,
            'answer': self.answer,
            'score': self.score,
            'used_count': self.used_count,
            'success_count': self.success_count,
            'fail_count': self.fail_count,
            'is_example': self.is_example,
            'source_users': self.source_users,
            'reference_count': self.reference_count,
            'last_contributor': self.last_contributor,
            'last_used': self.last_used
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            question=data['question'],
            answer=data['answer'],
            score=data.get('score', 0),
            used_count=data.get('used_count', 0),
            success_count=data.get('success_count', 0),
            fail_count=data.get('fail_count', 0),
            is_example=data.get('is_example', False),
            last_used=data.get('last_used', datetime.now().timestamp()),
            source_users=data.get('source_users', []),
            reference_count=data.get('reference_count', 0),
            last_contributor=data.get('last_contributor', '')
        )


class ModerateMemory:
    """适度记忆管理器 - 不让记忆无限膨胀"""
    
    def __init__(self, max_normal: int = 500, max_emergency: int = 1000):
        self.max_normal = max_normal      # 日常上限
        self.max_emergency = max_emergency  # 紧急上限
        self.current_limit = max_normal
        self.knowledge_items: List[KnowledgeItem] = []
        self.compression_count = 0  # 压缩次数统计
    
    def _calculate_similarity(self, q1: str, q2: str) -> float:
        """计算两个问题的相似度"""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return 0
        common = words1 & words2
        return len(common) / max(len(words1), len(words2))
    
    def _find_similar(self, item: KnowledgeItem, threshold: float = 0.7) -> List[KnowledgeItem]:
        """找到相似的知识条目"""
        similar = []
        for other in self.knowledge_items:
            if other is item:
                continue
            if self._calculate_similarity(item.question, other.question) > threshold:
                similar.append(other)
        return similar
    
    def _merge_similar(self):
        """合并相似知识"""
        merged_count = 0
        to_remove = []
        
        for item in self.knowledge_items:
            similar = self._find_similar(item, threshold=0.8)
            for sim in similar:
                if sim in to_remove:
                    continue
                # 合并：保留得分高的，合并统计
                if item.importance > sim.importance:
                    item.used_count += sim.used_count
                    item.success_count += sim.success_count
                    item.fail_count += sim.fail_count
                    item.reference_count = max(item.reference_count, sim.reference_count)
                    to_remove.append(sim)
                    merged_count += 1
                else:
                    sim.used_count += item.used_count
                    sim.success_count += item.success_count
                    sim.fail_count += item.fail_count
                    sim.reference_count = max(sim.reference_count, item.reference_count)
                    to_remove.append(item)
                    break
        
        for item in to_remove:
            if item in self.knowledge_items:
                self.knowledge_items.remove(item)
        
        if merged_count > 0:
            print(f"  🔄 记忆压缩：合并了{merged_count}条相似知识")
            self.compression_count += merged_count
    
    def _evict_low_importance(self, target_count: int):
        """淘汰低重要性知识"""
        if len(self.knowledge_items) <= target_count:
            return
        
        # 按重要性排序
        self.knowledge_items.sort(key=lambda x: x.importance)
        
        # 淘汰最不重要的
        remove_count = len(self.knowledge_items) - target_count
        removed = self.knowledge_items[:remove_count]
        self.knowledge_items = self.knowledge_items[remove_count:]
        
        print(f"  🗑️ 记忆清理：淘汰了{len(removed)}条低重要性知识")
        
        # 输出被淘汰的知识示例
        for r in removed[:3]:
            print(f"     - {r.question[:40]}... (重要性:{r.importance:.2f})")
    
    def _decay_low_frequency(self):
        """降权低频知识（不删除，只是更难被选中）"""
        decay_count = 0
        for item in self.knowledge_items:
            if item.used_count < 2 and not item.is_example:
                old_score = item.score
                item.score = max(-30, item.score * 0.7)
                if old_score != item.score:
                    decay_count += 1
        
        if decay_count > 0:
            print(f"  📉 记忆衰减：{decay_count}条低频知识降权")
    
    def add_knowledge(self, item: KnowledgeItem) -> bool:
        """添加知识，返回是否成功"""
        # 1. 检查是否已存在
        for existing in self.knowledge_items:
            if existing.question == item.question:
                print(f"⚠️ 知识已存在: {item.question[:30]}...")
                return False
        
        # 2. 检查相似度过高（合并）
        similar = self._find_similar(item, threshold=0.85)
        if similar:
            # 合并到最相似的
            best = similar[0]
            best.used_count += 1
            best.last_used = datetime.now().timestamp()
            print(f"🔗 知识已合并到相似条目: {item.question[:30]}...")
            return True
        
        # 3. 判断是否需要扩容
        if len(self.knowledge_items) >= self.current_limit:
            # 检查重要性是否足够突破限制
            if item.importance > 0.7:
                self.current_limit = min(self.max_emergency, self.current_limit + 50)
                print(f"  ⚡ 重要知识！临时扩容至{self.current_limit}")
            else:
                # 压缩现有记忆
                self._compress()
                
                # 如果还是满的，淘汰最不重要的
                if len(self.knowledge_items) >= self.current_limit:
                    self._evict_low_importance(self.current_limit - 1)
        
        # 4. 添加新知识
        self.knowledge_items.append(item)
        print(f"✅ 添加新知识: {item.question[:40]}... (重要性:{item.importance:.2f})")
        return True
    
    def _compress(self):
        """压缩记忆"""
        print(f"  🔧 执行记忆压缩 (当前:{len(self.knowledge_items)}/{self.current_limit})")
        
        # 1. 合并相似知识
        self._merge_similar()
        
        # 2. 降权低频知识
        self._decay_low_frequency()
        
        # 3. 如果还是超限，淘汰
        if len(self.knowledge_items) > self.current_limit:
            self._evict_low_importance(self.current_limit)
    
    def find_best_answer(self, question: str) -> Tuple[Optional[KnowledgeItem], float]:
        """找到最佳回答"""
        best_item = None
        best_score = -float('inf')
        
        norm_q = question.lower()
        
        for item in self.knowledge_items:
            # 匹配度
            match = self._calculate_similarity(question, item.question)
            # 总分 = 匹配度 * 10 + 得分 + 重要性*10
            score = match * 10 + item.score + item.importance * 10
            
            if score > best_score:
                best_score = score
                best_item = item
        
        if best_item:
            best_item.last_used = datetime.now().timestamp()
            best_item.used_count += 1
        
        return best_item, best_score
    
    def get_stats(self) -> dict:
        """获取记忆统计"""
        total_importance = sum(item.importance for item in self.knowledge_items)
        avg_importance = total_importance / len(self.knowledge_items) if self.knowledge_items else 0
        
        return {
            'total': len(self.knowledge_items),
            'limit': self.current_limit,
            'avg_importance': avg_importance,
            'high_importance': sum(1 for i in self.knowledge_items if i.importance > 0.7),
            'low_importance': sum(1 for i in self.knowledge_items if i.importance < 0.2),
            'compression_count': self.compression_count
        }
    
    def get_all(self) -> List[KnowledgeItem]:
        return self.knowledge_items
    
    def to_dict(self) -> dict:
        return {
            'items': [item.to_dict() for item in self.knowledge_items],
            'compression_count': self.compression_count
        }
    
    def from_dict(self, data: dict):
        self.knowledge_items = [KnowledgeItem.from_dict(d) for d in data.get('items', [])]
        self.compression_count = data.get('compression_count', 0)


# ============================================================
# 模仿学习模块
# ============================================================

class ImitationModule:
    def __init__(self):
        self.phrase_memory: Dict[str, int] = {}
        self.user_style: Dict[str, LanguageStyle] = {}
        self.user_phrases: Dict[str, List[str]] = {}
        self.imitation_count = 0
    
    def observe(self, user_id: str, user_text: str):
        words = user_text.split()
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            if len(word_clean) > 1:
                self.phrase_memory[word_clean] = self.phrase_memory.get(word_clean, 0) + 1
        
        if user_id not in self.user_phrases:
            self.user_phrases[user_id] = []
        self.user_phrases[user_id].append(user_text)
        if len(self.user_phrases[user_id]) > 50:
            self.user_phrases[user_id].pop(0)
        
        self._analyze_style(user_id, user_text)
        self.imitation_count += 1
    
    def _analyze_style(self, user_id: str, text: str):
        formal_words = ["请问", "您好", "感谢", "谢谢", "麻烦", "是否"]
        casual_words = ["咋", "啥", "呗", "啦", "哈", "嗯嗯", "哦"]
        friendly_words = ["嗨", "哈喽", "好哒", "没问题", "谢谢啦"]
        
        formal_score = sum(1 for w in formal_words if w in text)
        casual_score = sum(1 for w in casual_words if w in text)
        friendly_score = sum(1 for w in friendly_words if w in text)
        
        if formal_score > casual_score and formal_score > friendly_score:
            self.user_style[user_id] = LanguageStyle.FORMAL
        elif friendly_score > formal_score and friendly_score > casual_score:
            self.user_style[user_id] = LanguageStyle.FRIENDLY
        elif casual_score > formal_score:
            self.user_style[user_id] = LanguageStyle.CASUAL
        else:
            self.user_style[user_id] = LanguageStyle.NEUTRAL
    
    def get_style(self, user_id: str) -> LanguageStyle:
        return self.user_style.get(user_id, LanguageStyle.NEUTRAL)
    
    def get_common_phrases(self, user_id: str, n: int = 5) -> List[str]:
        return self.user_phrases.get(user_id, [])[-n:]
    
    def generate_imitated_response(self, user_id: str, intent: str, context: dict) -> Optional[str]:
        style = self.get_style(user_id)
        user_phrases = self.get_common_phrases(user_id, 3)
        
        if user_phrases:
            template = user_phrases[-1]
            if "?" in template or "？" in template:
                return template
            return f"{template}，{intent}？"
        
        if style == LanguageStyle.FORMAL:
            return f"请问您是想咨询{intent}吗？"
        elif style == LanguageStyle.CASUAL:
            return f"您是想问{intent}是吧？"
        elif style == LanguageStyle.FRIENDLY:
            return f"嗨！您是想了解{intent}吗？"
        else:
            return f"您需要{intent}吗？"
    
    def export(self) -> dict:
        return {
            'phrase_memory': self.phrase_memory,
            'user_style': {k: v.value for k, v in self.user_style.items()},
            'user_phrases': self.user_phrases
        }
    
    def import_data(self, data: dict):
        self.phrase_memory = data.get('phrase_memory', {})
        self.user_style = {k: LanguageStyle(v) for k, v in data.get('user_style', {}).items()}
        self.user_phrases = data.get('user_phrases', {})


# ============================================================
# 语言小AI
# ============================================================

class LanguageSmallAI:
    def __init__(self, style: LanguageStyle):
        self.style = style
        self.name = f"LangAI_{style.value}"
        self.fitness = 0
    
    def generate_response(self, intent: str, context: dict) -> str:
        if self.style == LanguageStyle.FORMAL:
            return f"请问您是想咨询{intent}吗？"
        elif self.style == LanguageStyle.CASUAL:
            return f"您是想问{intent}是吧？"
        elif self.style == LanguageStyle.FRIENDLY:
            return f"嗨！您是想了解{intent}吗？"
        elif self.style == LanguageStyle.PROFESSIONAL:
            return f"关于{intent}，请问您需要什么帮助？"
        else:
            return f"您需要{intent}吗？"


class LanguagePopulation:
    def __init__(self):
        self.language_ais: List[LanguageSmallAI] = []
        self.best_ai: Optional[LanguageSmallAI] = None
        self._init_population()
    
    def _init_population(self):
        for style in LanguageStyle:
            self.language_ais.append(LanguageSmallAI(style))
    
    def select_best(self, user_id: str, imitation: ImitationModule) -> LanguageSmallAI:
        user_style = imitation.get_style(user_id)
        
        for ai in self.language_ais:
            if ai.style == user_style:
                self.best_ai = ai
                return ai
        
        for ai in self.language_ais:
            if ai.style == LanguageStyle.NEUTRAL:
                self.best_ai = ai
                return ai
        
        self.best_ai = self.language_ais[0]
        return self.best_ai
    
    def get_best_response(self, user_id: str, imitation: ImitationModule, intent: str, context: dict) -> str:
        best_ai = self.select_best(user_id, imitation)
        
        imitated = imitation.generate_imitated_response(user_id, intent, context)
        if imitated:
            return imitated
        
        return best_ai.generate_response(intent, context)


# ============================================================
# 记忆系统（网格世界用）
# ============================================================

@dataclass
class MemoryItem2D:
    type: MemoryType
    x: int
    y: int
    action: str = None
    reward: float = 0
    confidence: float = 0.3
    count: int = 1
    last_used: float = field(default_factory=lambda: datetime.now().timestamp())
    is_negative: bool = False


class MemorySystem2D:
    def __init__(self, ai_id: int):
        self.ai_id = ai_id
        self.rules: List[MemoryItem2D] = []
        self.mutations: List[MemoryItem2D] = []
        self.warnings: List[MemoryItem2D] = []
        self.time = 0
        self.recent_events = []
    
    def _is_repetitive(self, x: int, y: int, action: str) -> bool:
        for event in self.recent_events[-10:]:
            if event.get('x') == x and event.get('y') == y and event.get('action') == action:
                return True
        return False
    
    def _find_rule(self, x: int, y: int, action: str) -> Optional[MemoryItem2D]:
        for rule in self.rules:
            if rule.x == x and rule.y == y and rule.action == action:
                return rule
        return None
    
    def add_event(self, x: int, y: int, action: str, reward: float):
        self.time += 1
        
        if reward <= -40:
            mutation = MemoryItem2D(
                type=MemoryType.MUTATION,
                x=x, y=y, action=action,
                reward=reward, confidence=1.0,
                is_negative=True
            )
            self.mutations.append(mutation)
            
            if action == 'right' and x > 0:
                self._add_warning(x-1, y, 'right')
            elif action == 'left' and x < 9:
                self._add_warning(x+1, y, 'left')
            elif action == 'up' and y > 0:
                self._add_warning(x, y-1, 'up')
            elif action == 'down' and y < 9:
                self._add_warning(x, y+1, 'down')
            return
        
        if self._is_repetitive(x, y, action):
            rule = self._find_rule(x, y, action)
            if rule:
                rule.confidence = min(1.0, rule.confidence + 0.1)
                rule.count += 1
                rule.last_used = self.time
            return
        
        if abs(reward) > 0.5:
            new_rule = MemoryItem2D(
                type=MemoryType.RULE,
                x=x, y=y, action=action,
                reward=reward, confidence=0.3
            )
            self.rules.append(new_rule)
        
        self.recent_events.append({'x': x, 'y': y, 'action': action})
        if len(self.recent_events) > 30:
            self.recent_events.pop(0)
        
        if self.time % 30 == 0:
            self._forget()
    
    def _add_warning(self, x: int, y: int, action: str):
        warning = MemoryItem2D(
            type=MemoryType.WARNING,
            x=x, y=y, action=action,
            reward=-50, confidence=0.8,
            is_negative=True
        )
        self.warnings.append(warning)
    
    def _forget(self):
        now = self.time
        before = len(self.rules)
        self.rules = [r for r in self.rules if now - r.last_used < 100]
        self.mutations = [m for m in self.mutations if now - m.last_used < 300]
        self.warnings = [w for w in self.warnings if now - w.last_used < 150]
    
    def is_dangerous(self, x: int, y: int, action: str) -> Tuple[bool, float]:
        target_x, target_y = x, y
        if action == 'up': target_y -= 1
        elif action == 'down': target_y += 1
        elif action == 'left': target_x -= 1
        elif action == 'right': target_x += 1
        
        for m in self.mutations:
            if m.x == target_x and m.y == target_y and m.action == action:
                return True, m.reward
        
        for w in self.warnings:
            if w.x == x and w.y == y and w.action == action:
                return True, w.reward
        
        return False, 0
    
    def get_expected_reward(self, x: int, y: int, action: str) -> Optional[float]:
        for rule in self.rules:
            if rule.x == x and rule.y == y and rule.action == action:
                rule.last_used = self.time
                return rule.reward * rule.confidence
        return None
    
    def export(self) -> dict:
        return {
            'rules': [(r.x, r.y, r.action, r.reward, r.confidence, r.count) for r in self.rules],
            'mutations': [(m.x, m.y, m.action, m.reward) for m in self.mutations],
            'warnings': [(w.x, w.y, w.action) for w in self.warnings]
        }
    
    def import_memories(self, data: dict):
        for x, y, action, reward, confidence, count in data.get('rules', []):
            self.rules.append(MemoryItem2D(
                type=MemoryType.RULE, x=x, y=y, action=action,
                reward=reward, confidence=confidence, count=count
            ))
        for x, y, action, reward in data.get('mutations', []):
            self.mutations.append(MemoryItem2D(
                type=MemoryType.MUTATION, x=x, y=y, action=action,
                reward=reward, confidence=1.0, is_negative=True
            ))


# ============================================================
# 简化版大AI（用于客服演示）
# ============================================================

class CustomerServiceAI:
    def __init__(self, ai_id: int = 0):
        self.id = ai_id
        self.memory = ModerateMemory(max_normal=200, max_emergency=400)  # 适度记忆
        self.imitation = ImitationModule()
        self.language_population = LanguagePopulation()
        self.conversation_history = []
    
    def chat(self, user_id: str, user_message: str) -> str:
        """客服对话接口"""
        self.imitation.observe(user_id, user_message)
        
        intent = self._parse_intent(user_message)
        
        # 在记忆库中查找
        best_item, score = self.memory.find_best_answer(user_message)
        
        if best_item:
            response = self.language_population.get_best_response(
                user_id, self.imitation, intent, {'question': user_message}
            )
            return f"{response} {best_item.answer}"
        else:
            return f"抱歉，我还没学过这个问题。您能教我怎么回答吗？"
    
    def teach(self, question: str, answer: str, user_id: str = None) -> bool:
        """教AI新知识"""
        new_item = KnowledgeItem(question=question, answer=answer)
        if user_id:
            new_item.add_contributor(user_id)
        return self.memory.add_knowledge(new_item)
    
    def feedback(self, question: str, is_helpful: bool, user_id: str = None):
        """用户反馈"""
        best_item, _ = self.memory.find_best_answer(question)
        if best_item:
            if is_helpful:
                best_item.score += 10
                best_item.success_count += 1
                if user_id:
                    best_item.add_contributor(user_id)
                print(f"👍 感谢反馈！这条知识得分+10")
            else:
                best_item.score -= 20
                best_item.fail_count += 1
                print(f"👎 已记录，这条知识得分-20")
    
    def _parse_intent(self, message: str) -> str:
        if "多少钱" in message or "价格" in message:
            return "价格咨询"
        elif "退货" in message or "退款" in message:
            return "售后问题"
        elif "注册" in message or "账号" in message:
            return "账号问题"
        elif "你好" in message or "您好" in message:
            return "问候"
        else:
            return "一般咨询"
    
    def get_memory_stats(self) -> dict:
        return self.memory.get_stats()


# ============================================================
# 客服演示
# ============================================================

def demo_customer_service():
    print("\n" + "="*60)
    print("🤖 AI客服演示 - 适度记忆版")
    print("="*60)
    print("\n【记忆机制】")
    print("1. 知识库有上限（日常200条，紧急400条）")
    print("2. 相似知识会自动合并")
    print("3. 低频知识会被降权")
    print("4. 低重要性知识会被淘汰")
    print("5. 重要知识可以临时扩容\n")
    
    ai = CustomerServiceAI()
    
    # 添加一些初始知识
    print("📚 初始化知识库...")
    ai.teach("产品多少钱", "99元起")
    ai.teach("怎么退货", "7天内无理由退货")
    ai.teach("怎么注册", "点击注册按钮即可")
    ai.teach("客服电话", "400-123-4567")
    
    print("\n开始对话（输入命令：/stats 查看记忆统计，/quit 退出）\n")
    
    user_id = "test_user"
    
    while True:
        user_input = input("\n👤 你: ").strip()
        
        if user_input.lower() == '/quit':
            print("👋 再见！")
            break
        
        if user_input.lower() == '/stats':
            stats = ai.get_memory_stats()
            print(f"\n📊 记忆统计:")
            print(f"   总知识数: {stats['total']}/{stats['limit']}")
            print(f"   平均重要性: {stats['avg_importance']:.2f}")
            print(f"   高重要性知识: {stats['high_importance']}条")
            print(f"   低重要性知识: {stats['low_importance']}条")
            print(f"   压缩次数: {stats['compression_count']}")
            continue
        
        if user_input.lower().startswith('/teach '):
            # 教AI新知识: /teach 问题 | 答案
            parts = user_input[7:].split('|')
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip()
                ai.teach(question, answer, user_id)
                print(f"✅ 已学习：{question} -> {answer}")
            else:
                print("格式：/teach 问题 | 答案")
            continue
        
        # 正常对话
        response = ai.chat(user_id, user_input)
        print(f"🤖 AI: {response}")
        
        # 询问反馈
        fb = input("这个回答有用吗？(y/n/skip): ").strip().lower()
        if fb == 'y':
            ai.feedback(user_input, True, user_id)
        elif fb == 'n':
            ai.feedback(user_input, False, user_id)
        
        # 每10轮显示一次统计
        if len(ai.conversation_history) % 10 == 0 and len(ai.conversation_history) > 0:
            stats = ai.get_memory_stats()
            print(f"\n📊 [系统] 当前记忆: {stats['total']}/{stats['limit']}")


# ============================================================
# 主程序
# ============================================================

def run():
    print("="*60)
    print("🧠 AI客服系统 - 适度记忆版")
    print("="*60)
    print("\n【核心特性】")
    print("1. 记忆不会无限膨胀（日常200条，紧急400条）")
    print("2. 相似知识自动合并")
    print("3. 低频知识自动降权")
    print("4. 低重要性知识自动淘汰")
    print("5. 重要知识可临时扩容")
    print("6. 模仿用户说话风格")
    print("7. 语言小AI竞争\n")
    
    demo_customer_service()


if __name__ == "__main__":
    run()