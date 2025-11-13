# real_memory_test.py
import psutil
import time
import threading
import requests
import json
from datetime import datetime

class RealMemoryTest:
    def __init__(self):
        self.process = None
        self.memory_data = []
        self.test_results = []
        
    def find_app_process(self):
        """找到运行的app.py进程"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('app.py' in cmd for cmd in proc.info['cmdline']):
                    print(f"✅ 找到应用进程: PID {proc.info['pid']}")
                    return psutil.Process(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        print("❌ 未找到运行的app.py进程")
        return None
    
    def record_memory(self, stage="baseline"):
        """记录内存使用"""
        if not self.process:
            return 0
            
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            record = {
                'timestamp': datetime.now().isoformat(),
                'memory_mb': memory_mb,
                'stage': stage
            }
            self.memory_data.append(record)
            return memory_mb
        except psutil.NoSuchProcess:
            return 0
    
    def trigger_real_prediction(self, text, test_id):
        """真正触发预测功能"""
        try:
            # 使用Gradio的API端点
            # 注意：Gradio 3.x 的API端点是 /api/predict/
            response = requests.post(
                "http://localhost:7860/api/predict/",
                json={
                    "data": [text],
                    "fn_index": 0  # 第一个函数，通常是预测函数
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ 预测成功: {text[:30]}...")
                return True
            else:
                print(f"❌ 预测失败 {response.status_code}: {text[:30]}...")
                return False
                
        except Exception as e:
            print(f"❌ 请求异常: {e}")
            return False
    
    def run_direct_function_test(self):
        """直接调用函数进行测试"""
        print("🔧 直接函数调用测试模式")
        
        # 导入你的系统模块
        try:
            # 重新导入系统模块以直接调用函数
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            
            # 重新初始化系统（这会触发真实的内存使用）
            print("🔄 重新初始化系统...")
            memory_before = self.record_memory("before_reinit")
            
            # 这里我们无法直接重新初始化，所以通过其他方式触发内存使用
            # 我们将通过监控现有进程的内存变化来测试
            
        except Exception as e:
            print(f"⚠️ 直接导入失败: {e}")
    
    def run_manual_test_with_instructions(self):
        """手动测试模式，提供详细指导"""
        print("🎯 真实内存测试 - 手动模式")
        print("=" * 60)
        print("请按照以下步骤操作，我会监控内存变化:")
        print()
        
        # 测试用例
        test_cases = [
            "楼道灯坏了三天，晚上回家很不方便，能尽快修一下吗？",
            "地下车库有陌生人员徘徊，感觉不太安全，请加强巡逻", 
            "我家老人突然头晕呕吐，急需医疗帮助！",
            "保洁阿姨打扫得很干净，楼道一尘不染，为你们的辛勤付出点赞！",
            "电梯经常故障，上周被困了十分钟，太吓人了"
        ]
        
        input("第一步: 确保应用正在运行，按 Enter 开始基准测试...")
        
        # 基准内存测试
        print("📊 记录基准内存...")
        baseline_readings = []
        for i in range(10):
            memory = self.record_memory("baseline")
            baseline_readings.append(memory)
            print(f"  基准 {i+1}/10: {memory:.1f} MB")
            time.sleep(1)
        
        baseline_avg = sum(baseline_readings) / len(baseline_readings)
        print(f"📈 平均基准内存: {baseline_avg:.1f} MB")
        
        # 开始真实测试
        print("\n第二步: 开始真实预测测试")
        print("请在浏览器中打开: http://localhost:7860")
        print("在'智能工单分析'标签页中进行以下操作:")
        
        for i, text in enumerate(test_cases):
            print(f"\n--- 测试 {i+1}/{len(test_cases)} ---")
            print(f"📝 请输入文本: {text}")
            
            input("准备好后按 Enter 开始记录内存...")
            
            # 记录预测前内存
            memory_before = self.record_memory(f"test_{i}_before")
            print(f"📊 预测前内存: {memory_before:.1f} MB")
            
            print("🖱️  请点击'智能分析'按钮...")
            input("分析完成后按 Enter 记录内存变化...")
            
            # 记录预测后内存
            memory_after = self.record_memory(f"test_{i}_after")
            memory_used = memory_after - memory_before
            
            print(f"📈 预测后内存: {memory_after:.1f} MB")
            print(f"📊 内存变化: +{memory_used:.1f} MB")
            
            self.test_results.append({
                'test_id': i,
                'text': text,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_used': memory_used
            })
            
            # 等待内存稳定
            print("⏳ 等待内存稳定...")
            time.sleep(3)
        
        # 最终内存测试
        print("\n第三步: 最终内存测试")
        final_readings = []
        for i in range(5):
            memory = self.record_memory("final")
            final_readings.append(memory)
            print(f"  最终 {i+1}/5: {memory:.1f} MB")
            time.sleep(2)
        
        # 生成报告
        self.generate_report(baseline_avg, final_readings)
    
    def generate_report(self, baseline_avg, final_readings):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("📊 真实内存测试报告")
        print("=" * 60)
        
        # 计算统计数据
        peak_memory = max([m['memory_mb'] for m in self.memory_data])
        final_avg = sum(final_readings) / len(final_readings)
        
        # 计算预测时的平均内存增量
        prediction_increases = [r['memory_used'] for r in self.test_results if r['memory_used'] > 0]
        avg_prediction_increase = sum(prediction_increases) / len(prediction_increases) if prediction_increases else 0
        
        print(f"📈 基准内存: {baseline_avg:.1f} MB")
        print(f"🚀 峰值内存: {peak_memory:.1f} MB") 
        print(f"📊 最终内存: {final_avg:.1f} MB")
        print(f"🔍 平均预测内存增量: {avg_prediction_increase:.1f} MB")
        print(f"🧪 完成测试次数: {len(self.test_results)} 次")
        
        # 部署建议
        base_requirement = peak_memory
        safety_buffer = base_requirement * 0.3  # 30%安全缓冲
        system_overhead = 200  # 系统开销
        
        recommended_mb = base_requirement + safety_buffer + system_overhead
        recommended_gb = recommended_mb / 1024
        
        print(f"\n💡 部署配置建议:")
        print(f"├── 基础需求: {int(base_requirement)} MB")
        print(f"├── 安全缓冲: {int(safety_buffer)} MB (30%)")
        print(f"├── 系统开销: {system_overhead} MB")
        print(f"├── 推荐配置: {int(recommended_mb)} MB ({recommended_gb:.1f} GB)")
        print(f"└── 生产配置: {int(recommended_mb * 1.5)} MB ({recommended_gb * 1.5:.1f} GB)")
        
        # 保存详细数据
        self.save_detailed_data()
        
        print(f"\n💾 详细数据已保存: real_memory_test_report.json")
    
    def save_detailed_data(self):
        """保存详细测试数据"""
        report = {
            'summary': {
                'baseline_memory_mb': self.memory_data[0]['memory_mb'] if self.memory_data else 0,
                'peak_memory_mb': max([m['memory_mb'] for m in self.memory_data]),
                'recommended_memory_mb': 0,
                'test_count': len(self.test_results)
            },
            'memory_timeline': self.memory_data,
            'test_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # 计算推荐配置
        peak_memory = report['summary']['peak_memory_mb']
        report['summary']['recommended_memory_mb'] = int(peak_memory * 1.3 + 200)
        
        with open('real_memory_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def run(self):
        """运行测试"""
        print("🏠 社区智理平台 - 真实内存测试")
        print("⚠️  请确保应用已启动: python app.py")
        
        # 查找进程
        self.process = self.find_app_process()
        if not self.process:
            print("❌ 请先启动应用: python app.py")
            return
        
        print("📍 应用访问: http://localhost:7860")
        print()
        
        # 运行手动测试
        self.run_manual_test_with_instructions()

def main():
    """主函数"""
    tester = RealMemoryTest()
    tester.run()

if __name__ == "__main__":
    main()