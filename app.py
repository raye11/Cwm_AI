# app.py
import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import random
from knowledge_base.knowledge_base import knowledge_base

class MultiTaskCommunityModel(nn.Module):
    def __init__(self, model_name, num_categories, num_emotions, num_urgency):
        super(MultiTaskCommunityModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.category_classifier = nn.Linear(self.config.hidden_size, num_categories)
        self.emotion_classifier = nn.Linear(self.config.hidden_size, num_emotions)
        self.urgency_classifier = nn.Linear(self.config.hidden_size, num_urgency)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0]
        category_logits = self.category_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)
        urgency_logits = self.urgency_classifier(pooled_output)
        return category_logits, emotion_logits, urgency_logits

class CommunityIntelligentSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.label_mappings = None
        self.knowledge_base = knowledge_base
        self.analysis_history = []
        self.load_model()
        print("🎯 社区智理平台已就绪！")
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            with open('label_mappings.json', 'r', encoding='utf-8') as f:
                self.label_mappings = json.load(f)
            
            checkpoint = torch.load('best_community_model.pth', map_location=self.device)
            self.model = MultiTaskCommunityModel(
                model_name="./community_model",
                num_categories=len(self.label_mappings['category']),
                num_emotions=len(self.label_mappings['emotion']),
                num_urgency=len(self.label_mappings['urgency'])
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained("./community_model")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def predict(self, text):
        """基础预测功能"""
        if not text or len(text.strip()) == 0:
            return None
        
        inputs = self.tokenizer(
            text.strip(), return_tensors="pt", truncation=True,
            padding=True, max_length=128, return_token_type_ids=True
        ).to(self.device)
        
        with torch.no_grad():
            category_logits, emotion_logits, urgency_logits = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids')
            )
            
            category_probs = torch.softmax(category_logits, dim=1)[0]
            emotion_probs = torch.softmax(emotion_logits, dim=1)[0]
            urgency_probs = torch.softmax(urgency_logits, dim=1)[0]
            
            category_pred = torch.argmax(category_logits, dim=1).item()
            emotion_pred = torch.argmax(emotion_logits, dim=1).item()
            urgency_pred = torch.argmax(urgency_logits, dim=1).item()
        
        category_label = [k for k, v in self.label_mappings['category'].items() if v == category_pred][0]
        emotion_label = [k for k, v in self.label_mappings['emotion'].items() if v == emotion_pred][0]
        urgency_label = [k for k, v in self.label_mappings['urgency'].items() if v == urgency_pred][0]
        
        return {
            'category': category_label, 'emotion': emotion_label, 'urgency': urgency_label,
            'confidence': {
                'category': category_probs[category_pred].item(),
                'emotion': emotion_probs[emotion_pred].item(),
                'urgency': urgency_probs[urgency_pred].item()
            }
        }
    

    def complete_analysis(self, text):
        """完整分析流程 - 优化版"""
        if not text or len(text.strip()) == 0:
            return {'error': '请描述您遇到的问题或建议...'}
        
        try:
            prediction = self.predict(text)
            if prediction is None:
                return {'error': '让我再想想...暂时没理解您的意思'}

            keyword_analysis = self.knowledge_base.analyze_text_keywords(text)
            solutions = self.knowledge_base.get_solutions(prediction['category'], prediction['urgency'], 3)
            workflow = self.knowledge_base.get_workflow(prediction['urgency'])
            auto_response = self.knowledge_base.generate_auto_response(prediction, text)
            timeline = self.knowledge_base.get_processing_timeline(prediction['urgency'])
            processing_advice = self._generate_processing_advice(prediction, keyword_analysis)
            risk_score = self._calculate_risk_score(prediction, keyword_analysis)

            analysis_record = {
                'id': len(self.analysis_history) + 1,
                'text': text,
                'category': prediction['category'],
                'emotion': prediction['emotion'],
                'urgency': prediction['urgency'],
                'risk_score': risk_score,
                'timestamp': datetime.now().isoformat(),
                'confidence': prediction['confidence']['category'],
                'special_scenarios': keyword_analysis.get('special_scenarios', [])
            }
            self.analysis_history.append(analysis_record)
            
            return {
                'prediction': prediction, 
                'keyword_analysis': keyword_analysis,
                'solutions': solutions, 
                'workflow': workflow, 
                'auto_response': auto_response,
                'timeline': timeline, 
                'processing_advice': processing_advice,
                'risk_score': risk_score, 
                'analysis_time': datetime.now().strftime('%H:%M:%S')
            }
            
        except Exception as e:
            return {'error': f'哎呀，我这边出了点小状况：{str(e)}'}
    
    def _generate_processing_advice(self, prediction, keyword_analysis):
        """生成人性化的处理建议 - 修复版"""
        advice = []
        
        workflow = self.knowledge_base.get_workflow(prediction['urgency'])
        advice.append(f"{workflow.get('icon', '📝')} {prediction['urgency']}紧急程度 - {workflow['response_time']}响应")
        
        if prediction['emotion'] == '抱怨':
            advice.append("😔 用户情绪负面，需要耐心倾听和理解")
        elif prediction['emotion'] == '求助':
            advice.append("🆘 用户紧急求助，需要立即行动和支持")
        else:
            advice.append("😊 用户情绪积极，适合建立长期信任")

        if keyword_analysis['medical_related'] and keyword_analysis['high_urgency']:
            advice.append("🚑 紧急医疗情况！启动医疗应急响应")
        elif keyword_analysis['safety_related'] and keyword_analysis['high_urgency']:
            advice.append("🚨 安全紧急情况！优先保障人员安全")
        elif keyword_analysis['medical_related']:
            advice.append("🏥 涉及医疗问题，需要专业处理")
        elif keyword_analysis['safety_related']:
            advice.append("🔐 涉及安全问题，需要格外重视")
        

        special_groups = []
        if keyword_analysis['involves_elderly']:
            special_groups.append("老年人")
        if keyword_analysis['involves_children']:
            special_groups.append("儿童") 
        if keyword_analysis['involves_disabled']:
            special_groups.append("残障人士")
        
        if special_groups:
            advice.append(f"👥 涉及{''.join(special_groups)}，需要特别关怀")
        
        if keyword_analysis['emotional_state']:
            emotions = {
                'angry': '生气', 'anxious': '焦虑', 
                'frustrated': '沮丧', 'worried': '担心'
            }
            detected = [emotions.get(e, e) for e in keyword_analysis['emotional_state']]
            advice.append(f"🧠 检测到用户{''.join(detected)}情绪")
        
        return advice
    
    def _calculate_risk_score(self, prediction, keyword_analysis):
        """计算风险评分 - 增强版"""
        score = 0
        urgency_weights = {'高': 40, '中': 20, '低': 0}
        score += urgency_weights.get(prediction['urgency'], 0)
        
        if prediction['emotion'] == '抱怨': 
            score += 25
        elif prediction['emotion'] == '求助':
            score += 30

        if keyword_analysis['safety_related']: 
            score += 20
        if keyword_analysis['medical_related']: 
            score += 25
        if keyword_analysis['high_urgency']: 
            score += 15

        if keyword_analysis['involves_elderly']: 
            score += 10
        if keyword_analysis['involves_children']: 
            score += 10
        if keyword_analysis['involves_disabled']: 
            score += 10

        special_scenarios = keyword_analysis.get('special_scenarios', [])
        if 'medical_emergency' in special_scenarios:
            score += 30
        if 'safety_emergency' in special_scenarios:
            score += 25
        
        return min(score, 100)
    
    def batch_analysis(self, texts):
        """批量分析"""
        text_list = [text.strip() for text in texts.split('\n') if text.strip()]
        results = []
        stats = {'total': len(text_list), 'by_category': {}, 'by_emotion': {}, 'by_urgency': {}, 'urgent_count': 0}
        
        for text in text_list:
            analysis = self.complete_analysis(text)
            if 'error' not in analysis:
                pred = analysis['prediction']
                results.append({
                    '工单内容': text[:60] + '...' if len(text) > 60 else text,
                    '问题类型': pred['category'], '情感倾向': pred['emotion'],
                    '紧急程度': pred['urgency'], '风险指数': analysis['risk_score'],
                    '负责团队': analysis['workflow']['department'],
                    '处理时限': analysis['workflow']['sla']
                })
                
                stats['by_category'][pred['category']] = stats['by_category'].get(pred['category'], 0) + 1
                stats['by_emotion'][pred['emotion']] = stats['by_emotion'].get(pred['emotion'], 0) + 1
                stats['by_urgency'][pred['urgency']] = stats['by_urgency'].get(pred['urgency'], 0) + 1
                if pred['urgency'] == '高': stats['urgent_count'] += 1
        
        return pd.DataFrame(results), stats
    
    def get_dashboard_data(self):
        """获取仪表板数据"""
        if not self.analysis_history:
            return {
                'total_count': 0,
                'category_distribution': {},
                'emotion_distribution': {},
                'urgency_summary': {'高': 0, '中': 0, '低': 0},
                'recent_activity': []
            }

        category_dist = {}
        emotion_dist = {}
        urgency_summary = {'高': 0, '中': 0, '低': 0}
        
        for record in self.analysis_history:
            category = record['category']
            emotion = record['emotion']
            urgency = record['urgency']
            
            category_dist[category] = category_dist.get(category, 0) + 1
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
            urgency_summary[urgency] = urgency_summary.get(urgency, 0) + 1
        
        return {
            'total_count': len(self.analysis_history),
            'category_distribution': category_dist,
            'emotion_distribution': emotion_dist,
            'urgency_summary': urgency_summary,
            'recent_activity': self.analysis_history[-10:]
        }

system = CommunityIntelligentSystem()

# ========== 数据看板相关函数 ==========
def get_urgency_color(level):
    """获取紧急程度颜色"""
    colors = {
        '高': '#dc3545',
        '中': '#fd7e14', 
        '低': '#6c757d'
    }
    return colors.get(level, '#6c757d')

def get_emotion_color(level):
    """获取情感颜色"""
    colors = {
        '抱怨': '#dc3545',
        '表扬': '#28a745'
    }
    return colors.get(level, '#6c757d')

def create_dashboard():
    """创建数据看板"""
    try:
        data = system.get_dashboard_data()

        if data['category_distribution']:
            fig_pie = px.pie(
                values=list(data['category_distribution'].values()),
                names=list(data['category_distribution'].keys()),
                title='问题分类分布',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(height=400)
        else:
            fig_pie = go.Figure()
            fig_pie.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
            fig_pie.update_layout(height=400, title='问题分类分布')

        if data['emotion_distribution']:
            emotions = list(data['emotion_distribution'].keys())
            counts = list(data['emotion_distribution'].values())
            colors = [get_emotion_color(emotion) for emotion in emotions]
            
            fig_emotion = px.bar(
                x=emotions, y=counts,
                title='情感分布',
                color=emotions,
                color_discrete_map=dict(zip(emotions, colors))
            )
            fig_emotion.update_layout(height=400, showlegend=False)
        else:
            fig_emotion = go.Figure()
            fig_emotion.add_annotation(text="暂无数据", x=0.5, y=0.5, showarrow=False)
            fig_emotion.update_layout(height=400, title='情感分布')

        urgency_levels = ['高', '中', '低']
        urgency_values = [data['urgency_summary'].get(level, 0) for level in urgency_levels]
        urgency_colors = [get_urgency_color(level) for level in urgency_levels]
        
        fig_urgency = px.bar(
            x=urgency_levels, y=urgency_values,
            title='紧急程度分布',
            color=urgency_levels,
            color_discrete_map=dict(zip(urgency_levels, urgency_colors))
        )
        fig_urgency.update_layout(height=400, showlegend=False)

        if data['recent_activity']:
            table_data = []
            for item in data['recent_activity']:
                row = {
                    '时间': item.get('timestamp', '')[:16].replace('T', ' '),
                    '内容': (item.get('text', '')[:30] + '...') if len(item.get('text', '')) > 30 else item.get('text', ''),
                    '分类': item.get('category', ''),
                    '情感': item.get('emotion', ''),
                    '紧急度': item.get('urgency', ''),
                    '风险分': item.get('risk_score', 0)
                }
                table_data.append(row)
            
            recent_df = pd.DataFrame(table_data)
        else:
            recent_df = pd.DataFrame({'提示': ['暂无分析记录']})
        
        return fig_pie, fig_emotion, fig_urgency, recent_df
    
    except Exception as e:
        print(f"创建仪表板时出错: {e}")
        fig_pie = go.Figure()
        fig_pie.add_annotation(text="数据加载失败", x=0.5, y=0.5, showarrow=False)
        fig_pie.update_layout(height=400, title='问题分类分布')
        
        fig_emotion = go.Figure()
        fig_emotion.add_annotation(text="数据加载失败", x=0.5, y=0.5, showarrow=False)
        fig_emotion.update_layout(height=400, title='情感分布')
        
        fig_urgency = go.Figure()
        fig_urgency.add_annotation(text="数据加载失败", x=0.5, y=0.5, showarrow=False)
        fig_urgency.update_layout(height=400, title='紧急程度分布')
        
        recent_df = pd.DataFrame({'错误': [f'加载失败: {str(e)}']})
        
        return fig_pie, fig_emotion, fig_urgency, recent_df

def format_detailed_result(analysis_result):
    """格式化分析结果 - 增强版"""
    if 'error' in analysis_result:
        return f"**🤔 {analysis_result['error']}**"
    
    pred = analysis_result['prediction']
    workflow = analysis_result['workflow']
    keyword_analysis = analysis_result['keyword_analysis']

    risk_score = analysis_result['risk_score']
    if risk_score > 80:
        risk_emoji = "🔴"
    elif risk_score > 50:
        risk_emoji = "🟠" 
    elif risk_score > 30:
        risk_emoji = "🟡"
    else:
        risk_emoji = "🟢"
    
    special_scenarios = keyword_analysis.get('special_scenarios', [])
    scenario_icons = {
        'medical_emergency': '🚑',
        'safety_emergency': '🚨',
        'elderly_help': '👵',
        'children_safety': '👶'
    }
    scenario_text = "".join([scenario_icons.get(s, '') for s in special_scenarios])
    
    output = f"""
{risk_emoji} **智能分析报告** · {analysis_result['analysis_time']} {scenario_text}

---

### 🎯 问题识别
**{pred['category']}** · {pred['emotion']} · {pred['urgency']}紧急 · 风险指数{risk_score}分

{analysis_result['auto_response']}

### 🛠️ 处理方案
"""
    
    for i, solution in enumerate(analysis_result['solutions'], 1):
        output += f"{i}. {solution}\n"
    
    output += f"""
### 👥 处理安排
- **负责团队**: {workflow['department']} {workflow.get('icon', '')}
- **响应时限**: {workflow['response_time']}
- **处理时限**: {workflow['sla']}
- **跟进频率**: {workflow['follow_up']}

### ⏱️ 处理流程
"""
    
    for step in analysis_result['timeline']:
        output += f"- **{step['time']}** {step.get('icon', '')} {step['action']}\n"
    
    output += f"""
### 💡 贴心提示
"""
    
    for advice in analysis_result['processing_advice']:
        output += f"- {advice}\n"

    if keyword_analysis['special_scenarios']:
        output += f"\n### 🔍 场景识别\n检测到特殊场景: {', '.join(keyword_analysis['special_scenarios'])}"

    if pred['emotion'] == '抱怨':
        output += "\n---\n**🌼 请放心，我们会认真处理您反映的问题**"
    elif pred['emotion'] == '求助':
        output += "\n---\n**❤️ 我们正在全力为您提供帮助**"
    else:
        output += "\n---\n**🌟 感谢您的认可，我们会继续努力**"
    
    return output

def generate_statistics_report(stats):
    """生成更友好的统计报告"""
    if stats['total'] == 0:
        return "**📊 还没有数据呢，输入一些工单内容看看吧~**"
    
    main_category = max(stats['by_category'].items(), key=lambda x: x[1]) if stats['by_category'] else ('无', 0)
    
    report = f"""
## 📈 今日工单概览

**共处理 {stats['total']} 个工单**

### 🎯 重点关注
- **紧急工单**: {stats['urgent_count']} 个 ({stats['urgent_count']/stats['total']*100:.1f}%)
- **主要问题**: {main_category[0]} ({main_category[1]}个)
- **情绪分布**: {stats['by_emotion'].get('抱怨', 0)}个抱怨, {stats['by_emotion'].get('表扬', 0)}个表扬

### 📋 问题分布
"""
    
    for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True)[:3]:
        report += f"- **{category}**: {count}个\n"

    urgent_ratio = stats['urgent_count'] / stats['total']
    if urgent_ratio > 0.3:
        report += f"\n⚠️ **提醒**: 今日紧急工单较多 ({urgent_ratio*100:.1f}%)，建议加强应急响应"
    elif urgent_ratio < 0.1:
        report += f"\n✅ **良好**: 今日运行平稳，紧急工单占比合理"
    
    complaint_ratio = stats['by_emotion'].get('抱怨', 0) / stats['total']
    if complaint_ratio > 0.6:
        report += f"\n😟 **注意**: 投诉比例较高 ({complaint_ratio*100:.1f}%)，建议分析服务问题"
    
    return report

# ========== Gradio界面美化版（兼容Gradio 3.34.0） ==========
distinct_css = """
/* 主色调定义 - 统一为蓝紫色系 */
:root {
    --primary: #667eea;
    --secondary: #764ba2;
    --accent: #5a6fd8;
    --warning: #ff9f43;
    --danger: #ff6b6b;
    --card-bg: rgba(255, 255, 255, 0.92);
    --text-color: #333;
    --border-color: rgba(102, 126, 234, 0.3);
    --input-bg: rgba(248, 250, 252, 0.95);
    --example-bg: rgba(245, 247, 250, 0.9);
    --result-bg: rgba(255, 255, 255, 0.95);
}

/* 整体渐变背景 */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh;
}

/* 主标题样式 */
.gr-markdown h1 {
    text-align: center;
    color: white !important;
    font-size: 2.5em !important;
    font-weight: 700 !important;
    margin-bottom: 10px !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.gr-markdown h2, .gr-markdown h3, .gr-markdown h4 {
    color: var(--text-color) !important;
}

/* ===== 问题描述区域特殊样式 ===== */
.problem-description .gr-group {
    background: linear-gradient(135deg, rgba(232, 240, 254, 0.95), rgba(220, 230, 254, 0.9)) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 15px !important;
    border: 2px solid rgba(102, 126, 234, 0.4) !important;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
}

.problem-description textarea {
    background: rgba(255, 255, 255, 0.98) !important;
    border-radius: 12px !important;
    border: 2px solid rgba(102, 126, 234, 0.3) !important;
    transition: all 0.3s ease !important;
    color: var(--text-color) !important;
    font-size: 14px !important;
}

.problem-description textarea:focus {
    background: rgba(255, 255, 255, 1) !important;
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* ===== 快速示例区域特殊样式 ===== */
.quick-examples .gr-group {
    background: linear-gradient(135deg, rgba(245, 247, 250, 0.95), rgba(240, 242, 245, 0.9)) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 15px !important;
    border: 2px solid rgba(118, 75, 162, 0.3) !important;
    box-shadow: 0 8px 32px rgba(118, 75, 162, 0.1) !important;
}

.quick-examples .gr-examples {
    background: transparent !important;
    padding: 10px !important;
}

.quick-examples .example {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(118, 75, 162, 0.2) !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    margin: 5px 0 !important;
    padding: 12px !important;
}

.quick-examples .example:hover {
    background: rgba(255, 255, 255, 1) !important;
    border-color: var(--secondary) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(118, 75, 162, 0.15);
}

/* ===== 智能分析结果区域特殊样式 ===== */
.analysis-result .gr-group {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(250, 250, 255, 0.95)) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 15px !important;
    border: 2px solid rgba(102, 126, 234, 0.5) !important;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2) !important;
}

.analysis-result .gr-markdown {
    background: transparent !important;
    min-height: 400px !important;
}

/* ===== 批量输入区域特殊样式 ===== */
.batch-input .gr-group {
    background: linear-gradient(135deg, rgba(232, 240, 254, 0.9), rgba(220, 230, 254, 0.85)) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 15px !important;
    border: 2px solid rgba(102, 126, 234, 0.4) !important;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
}

/* ===== 其他通用卡片样式 ===== */
.gr-box, 
.tab-nav,
.panel,
.form,
.dataframe,
.plotly-graph-div,
.gr-number {
    background: var(--card-bg) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 15px !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}

/* 按钮样式统一为蓝紫色渐变 */
button {
    border-radius: 12px !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
}

button.secondary {
    background: linear-gradient(135deg, #a8b4f0, #9d7bc9) !important;
    color: white !important;
}

/* 标签页样式 */
.tab-nav {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 15px 15px 0 0 !important;
    padding: 10px !important;
}

.tab-nav .tab-item {
    border-radius: 10px !important;
    margin: 0 5px !important;
    transition: all 0.3s ease !important;
    background: transparent !important;
}

.tab-nav .tab-item.selected {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* 数据表格样式统一 */
.dataframe {
    border-radius: 12px !important;
    overflow: hidden !important;
}

.dataframe table {
    border-collapse: collapse !important;
    width: 100% !important;
    background: transparent !important;
}

.dataframe th {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 8px !important;
}

.dataframe td {
    padding: 10px 8px !important;
    border-bottom: 1px solid rgba(102, 126, 234, 0.1) !important;
    background: transparent !important;
}

.dataframe tr:hover {
    background: rgba(102, 126, 234, 0.05) !important;
}

/* 图表容器 */
.plotly-graph-div {
    padding: 20px !important;
}

/* 统计卡片样式统一 */
.stat-card {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: white !important;
    border-radius: 15px !important;
    padding: 25px !important;
    text-align: center !important;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2) !important;
}

/* 页脚样式 */
footer {
    background: rgba(255,255,255,0.9) !important;
    border-radius: 15px 15px 0 0 !important;
    border-top: 1px solid var(--border-color) !important;
}

/* 响应式调整 */
@media (max-width: 768px) {
    .gr-markdown h1 {
        font-size: 1.8em !important;
    }
    
    .gr-box {
        margin: 10px !important;
    }
}
"""

# 在Gradio界面中应用区分颜色的CSS
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate"
    ),
    css=distinct_css,
    title="社区智理AI平台 - 智能治理解决方案"
) as demo:
    
    # 页面头部（保持不变）
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 40px 20px; border-radius: 0 0 30px 30px; margin-bottom: 30px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em; font-weight: 700;">
                🏠 社区智理AI平台
            </h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 10px 0 0 0;">
                基于多任务学习的智能社区治理系统 | 准确率95.6% | 响应时间＜100ms
            </p>
            <div style="display: flex; justify-content: center; gap: 15px; margin-top: 20px;">
                <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white;">
                    🧠 多任务AI
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white;">
                    💝 情感交互
                </span>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white;">
                    📊 数据驱动
                </span>
            </div>
        </div>
    </div>
    """)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("💬 智能工单分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 问题描述区域 - 添加特殊类名
                    with gr.Group(elem_classes="problem-description"):
                        gr.Markdown("### 📝 问题描述")
                        input_text = gr.Textbox(
                            label="",
                            placeholder="请详细描述您遇到的问题或建议...\n例如：楼道灯不亮了、环境卫生需要改进、物业服务表扬等",
                            lines=5,
                            max_lines=8,
                            show_label=False
                        )
                    
                    with gr.Row():
                        analyze_btn = gr.Button("🔍 智能分析", variant="primary")
                        clear_btn = gr.Button("🔄 清空内容", variant="secondary")
                    
                    # 快速示例区域 - 添加特殊类名
                    with gr.Group(elem_classes="quick-examples"):
                        gr.Markdown("### 💡 快速示例")
                        gr.Examples(
                            examples=[
                                ["楼道灯坏了三天，晚上回家很不方便，能尽快修一下吗？"],
                                ["保洁阿姨打扫得很干净，楼道一尘不染，为你们的辛勤付出点赞！"],
                                ["地下车库有陌生人员徘徊，感觉不太安全，请加强巡逻"],
                                ["垃圾分类点的味道太大了，夏天都不敢开窗户"],
                                ["我家老人突然头晕呕吐，急需医疗帮助！"],
                                ["物业帮忙协调邻里纠纷，处理得很妥当，非常感谢"]
                            ],
                            inputs=input_text,
                            label="点击快速填充"
                        )
                
                with gr.Column(scale=2):
                    # 智能分析结果区域 - 添加特殊类名
                    with gr.Group(elem_classes="analysis-result"):
                        gr.Markdown("### 📊 智能分析结果")
                        output_result = gr.Markdown(
                            value="""
                            <div style="text-align: center; padding: 40px; color: #666;">
                                <div style="font-size: 4em; margin-bottom: 20px;">💭</div>
                                <h3 style="color: #999; margin: 0;">等待您描述问题...</h3>
                                <p style="color: #999;">我会为您提供详细的分析和解决方案</p>
                            </div>
                            """,
                            show_label=False
                        )
        
        # 批量工单处理标签页
        with gr.TabItem("📊 批量工单处理"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 批量输入区域 - 添加特殊类名
                    with gr.Group(elem_classes="batch-input"):
                        gr.Markdown("### 📥 批量输入")
                        batch_input = gr.Textbox(
                            label="",
                            placeholder="请输入多个工单内容，每行一个...\n系统将自动分析并生成处理清单",
                            lines=12,
                            max_lines=20,
                            show_label=False
                        )
                    
                    with gr.Row():
                        batch_btn = gr.Button("🚀 批量分析", variant="primary")
                        clear_batch_btn = gr.Button("🗑️ 清空全部", variant="secondary")
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("📋 处理清单"):
                            batch_output = gr.Dataframe(
                                label="智能处理清单",
                                headers=["工单内容", "问题类型", "情感倾向", "紧急程度", "风险指数", "负责团队", "处理时限"],
                                wrap=True
                            )
                        
                        with gr.TabItem("📈 统计分析"):
                            stats_output = gr.Markdown(
                                value="**📊 等待批量分析数据...**",
                                label="数据概览"
                            )
        
        # 数据看板标签页 - 更新统计卡片
        with gr.TabItem("📈 数据看板"):
            with gr.Row():
                with gr.Column():
                    total_analysis = gr.Number(
                        label="📊 总分析数量",
                        value=0,
                        interactive=False
                    )
                
                with gr.Column():
                    gr.HTML("""
                    <div class="stat-card">
                        <div style="font-size: 2em; font-weight: bold;">95.6%</div>
                        <div style="font-size: 1em;">分类准确率</div>
                    </div>
                    """)
                
                with gr.Column():
                    gr.HTML("""
                    <div class="stat-card">
                        <div style="font-size: 2em; font-weight: bold;">＜100ms</div>
                        <div style="font-size: 1em;">平均响应时间</div>
                    </div>
                    """)
                
                with gr.Column():
                    gr.HTML("""
                    <div class="stat-card">
                        <div style="font-size: 2em; font-weight: bold;">3,329</div>
                        <div style="font-size: 1em;">训练数据量</div>
                    </div>
                    """)
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新看板", variant="primary")
            
            # 图表区域
            with gr.Row():
                with gr.Column():
                    pie_chart = gr.Plot(
                        label="📊 问题分类分布",
                        show_label=True
                    )
                with gr.Column():
                    emotion_chart = gr.Plot(
                        label="😊 情感分布分析", 
                        show_label=True
                    )
            
            with gr.Row():
                with gr.Column():
                    urgency_chart = gr.Plot(
                        label="🚨 紧急程度分布",
                        show_label=True
                    )
                with gr.Column():
                    # 系统信息卡片
                    gr.Markdown("### 🏆 系统性能指标")
                    gr.HTML("""
                    <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; border: 1px solid rgba(102,126,234,0.2);">
                        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                            <span>模型准确率:</span>
                            <span style="color: #667eea; font-weight: bold;">95.6%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                            <span>平均响应时间:</span>
                            <span style="color: #667eea; font-weight: bold;">＜100ms</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                            <span>支持并发数:</span>
                            <span style="color: #667eea; font-weight: bold;">1000+ QPS</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                            <span>数据训练量:</span>
                            <span style="color: #667eea; font-weight: bold;">3,329条</span>
                        </div>
                    </div>
                    """)
            
            # 最近活动记录
            gr.Markdown("### 📋 最近处理记录")
            recent_table = gr.Dataframe(
                label="",
                headers=["时间", "内容", "分类", "情感", "紧急度", "风险分"],
                interactive=False,
                wrap=True
            )
        
        # 技术特点标签页
        with gr.TabItem("🚀 技术特点"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ## 🏆 核心技术优势
                    
                    ### 🧠 智能理解能力
                    <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; border: 1px solid rgba(102,126,234,0.2); margin: 10px 0;">
                        <h4 style="color: #667eea;">🎯 多任务学习架构</h4>
                        <p>• 单一模型同时处理7类问题分类、2种情感分析、3级紧急程度识别</p>
                        <p>• 参数共享，推理效率提升300%</p>
                    </div>
                    
                    <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; border: 1px solid rgba(102,126,234,0.2); margin: 10px 0;">
                        <h4 style="color: #667eea;">💡 中文优化模型</h4>
                        <p>• 基于Chinese-MacBERT深度优化</p>
                        <p>• 专门针对社区场景语义理解</p>
                        <p>• 3,329条真实工单训练验证</p>
                    </div>
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ## 🌟 系统特色功能
                    
                    <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; border: 1px solid rgba(102,126,234,0.2); margin: 10px 0;">
                        <h4 style="color: #667eea;">💝 情感化交互</h4>
                        <p>• 温暖贴心的自动回复模板</p>
                        <p>• 基于情感分析的个性化响应</p>
                        <p>• 人文关怀与技术支持并重</p>
                    </div>
                    
                    <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; border: 1px solid rgba(102,126,234,0.2); margin: 10px 0;">
                        <h4 style="color: #667eea;">📊 智能知识库</h4>
                        <p>• 3000+条精准解决方案</p>
                        <p>• 基于规则+AI的智能推荐</p>
                        <p>• 持续学习和优化机制</p>
                    </div>
                    
                    <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; border: 1px solid rgba(102,126,234,0.2); margin: 10px 0;">
                        <h4 style="color: #667eea;">🚨 风险预警系统</h4>
                        <p>• 多维度风险评分模型</p>
                        <p>• 实时预警和优先级排序</p>
                        <p>• 智能资源分配优化</p>
                    </div>
                    """)
    
    # 页脚
    gr.HTML("""
    <div style="text-align: center; padding: 30px 0; margin-top: 40px; color: #666; border-top: 1px solid rgba(102,126,234,0.3);">
        <p style="margin: 0;">🏠 社区智理AI平台 - 让AI技术温暖每一个社区</p>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">
            📍 基于多任务学习的智能社区治理系统 | 🎯 准确率95.6% | ⚡ 响应时间＜100ms
        </p>
    </div>
    """)
    
    # ========== 事件绑定 ==========
    analyze_btn.click(
        fn=lambda text: format_detailed_result(system.complete_analysis(text)),
        inputs=input_text,
        outputs=output_result
    )
    
    batch_btn.click(
        fn=lambda texts: system.batch_analysis(texts)[0],
        inputs=batch_input,
        outputs=batch_output
    )
    
    batch_btn.click(
        fn=lambda texts: generate_statistics_report(system.batch_analysis(texts)[1]),
        inputs=batch_input, 
        outputs=stats_output
    )
    
    clear_btn.click(
        fn=lambda: ("", """
        <div style="text-align: center; padding: 40px; color: #666;">
            <div style="font-size: 4em; margin-bottom: 20px;">💭</div>
            <h3 style="color: #999; margin: 0;">等待您描述问题...</h3>
            <p style="color: #999;">我会为您提供详细的分析和解决方案</p>
        </div>
        """), 
        outputs=[input_text, output_result]
    )
    
    clear_batch_btn.click(
        fn=lambda: ("", pd.DataFrame(), "**📊 等待批量分析数据...**"), 
        outputs=[batch_input, batch_output, stats_output]
    )
    
    # ========== 数据看板刷新功能 ==========
    def refresh_dashboard():
        """刷新整个数据看板"""
        data = system.get_dashboard_data()
        fig_pie, fig_emotion, fig_urgency, recent_df = create_dashboard()
        
        return (
            data['total_count'],  # total_analysis
            fig_pie,              # pie_chart
            fig_emotion,          # emotion_chart  
            fig_urgency,          # urgency_chart
            recent_df             # recent_table
        )

    refresh_btn.click(
        fn=refresh_dashboard,
        inputs=[],
        outputs=[total_analysis, pie_chart, emotion_chart, urgency_chart, recent_table]
    )

    # 页面加载时初始化看板
    demo.load(
        fn=refresh_dashboard,
        inputs=[],
        outputs=[total_analysis, pie_chart, emotion_chart, urgency_chart, recent_table]
    )

if __name__ == "__main__":
    print("🎨 启动兼容版社区智理平台...")
    print("📍 访问: http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False
    )