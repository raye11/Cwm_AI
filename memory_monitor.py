# memory_monitor.py
import psutil
import time
import threading
import requests
import json
from datetime import datetime
import pandas as pd

class MemoryMonitor:
    def __init__(self, process_name="python"):
        self.process_name = process_name
        self.memory_stats = {
            'startup': 0,
            'peak': 0,
            'predictions': [],
            'timestamps': [],
            'concurrent_peak': 0
        }
        self.current_concurrent = 0
        self.lock = threading.Lock()
        
    def find_app_process(self):
        """找到运行的app.py进程"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('app.py' in cmd for cmd in proc.info['cmdline']):
                    return psutil.Process(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def start_monitoring(self, interval=2):
        """开始监控内存使用"""
        self.process = self.find_app_process()
        if not self.process:
            print("❌ 未找到运行的app.py进程，请先启动应用")
            return False
            
        print(f"🔍 开始监控进程: {self.process.pid}")
        self.memory_stats['startup'] = self.record_memory("startup")
        
        def monitor_loop():
            while True:
                try:
                    if not self.process.is_running():
                        print("⚠️ 应用进程已停止")
                        break
                    self.record_memory("background")
                    time.sleep(interval)
                except psutil.NoSuchProcess:
                    print("⚠️ 进程不存在")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        return True
    
    def record_memory(self, event="runtime"):
        """记录内存使用"""
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            with self.lock:
                self.memory_stats['timestamps'].append({
                    'time': datetime.now().isoformat(),
                    'memory_mb': memory_mb,
                    'event': event,
                    'concurrent': self.current_concurrent
                })
                
                if memory_mb > self.memory_stats['peak']:
                    self.memory_stats['peak'] = memory_mb
                
                # 记录并发峰值
                if self.current_concurrent > 0 and memory_mb > self.memory_stats['concurrent_peak']:
                    self.memory_stats['concurrent_peak'] = memory_mb
            
            return memory_mb
        except psutil.NoSuchProcess:
            return 0
    
    def record_concurrent_start(self):
        """记录并发请求开始"""
        with self.lock:
            self.current_concurrent += 1
    
    def record_concurrent_end(self, memory_used):
        """记录并发请求结束"""
        with self.lock:
            self.current_concurrent -= 1
            self.memory_stats['predictions'].append(memory_used)
    
    def get_realtime_stats(self):
        """获取实时统计"""
        if not self.memory_stats['timestamps']:
            return "无数据"
            
        current = self.memory_stats['timestamps'][-1]['memory_mb']
        return f"当前: {current:.1f}MB | 峰值: {self.memory_stats['peak']:.1f}MB | 并发峰值: {self.memory_stats['concurrent_peak']:.1f}MB"
    
    def generate_deployment_report(self):
        """生成部署报告"""
        if not self.memory_stats['predictions']:
            return "**📊 等待收集测试数据...**"
        
        # 计算统计数据
        avg_prediction = sum(self.memory_stats['predictions']) / len(self.memory_stats['predictions'])
        max_prediction = max(self.memory_stats['predictions']) if self.memory_stats['predictions'] else 0
        
        # 内存需求计算
        base_memory = self.memory_stats['peak']
        safety_buffer = base_memory * 0.3  # 30%安全缓冲
        system_overhead = 200  # 系统开销
        
        recommended_mb = base_memory + safety_buffer + system_overhead
        
        report = f"""
🎯 **实际部署内存需求分析报告**

📊 **测试数据统计**
├── 启动内存: {self.memory_stats['startup']:.1f} MB
├── 峰值内存: {self.memory_stats['peak']:.1f} MB
├── 并发峰值: {self.memory_stats['concurrent_peak']:.1f} MB
├── 平均请求内存: {avg_prediction:.1f} MB
├── 最大请求内存: {max_prediction:.1f} MB
└── 总测试次数: {len(self.memory_stats['predictions'])} 次

💡 **部署配置建议**
├── 基础需求: {int(base_memory)} MB
├── 安全缓冲: {int(safety_buffer)} MB (30%)
├── 系统开销: {system_overhead} MB
├── **推荐配置**: {int(recommended_mb)} MB ({recommended_mb/1024:.1f} GB)
└── 生产环境: {int(recommended_mb * 1.5)} MB ({recommended_mb * 1.5 / 1024:.1f} GB)

🔧 **说明**
- 测试环境: 单机部署
- 并发用户: 模拟{self.memory_stats['timestamps'][-1]['concurrent'] if self.memory_stats['timestamps'] else 0}个
- 建议基于峰值内存 + 30%缓冲 + 系统开销
        """
        
        return report
    
    def save_detailed_report(self, filename="memory_report.json"):
        """保存详细报告"""
        report = {
            'summary': {
                'peak_memory_mb': self.memory_stats['peak'],
                'concurrent_peak_mb': self.memory_stats['concurrent_peak'],
                'recommended_memory_mb': self.memory_stats['peak'] * 1.3 + 200,
                'test_count': len(self.memory_stats['predictions'])
            },
            'detailed_data': self.memory_stats
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 详细报告已保存: {filename}")