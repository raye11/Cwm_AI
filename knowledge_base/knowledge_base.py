# knowledge.py
"""
社区智能管家知识库 - 全面增强版
在保持现有接口不变的基础上，大幅扩充知识库内容和智能程度
"""

class CommunityKnowledgeBase:
    def __init__(self):
        self.solution_knowledge = self._load_solution_knowledge()
        self.response_templates = self._load_response_templates()
        self.workflow_rules = self._load_workflow_rules()
        self.keyword_triggers = self._load_keyword_triggers()
        self.care_phrases = self._load_care_phrases()
        self.special_scenarios = self._load_special_scenarios()
        self.seasonal_knowledge = self._load_seasonal_knowledge()
        self.scenario_name_mapping = self._load_scenario_name_mapping()

    def _load_scenario_name_mapping(self):
        """加载场景名称中英文映射"""
        return {
            'medical_emergency': '医疗紧急情况',
            'safety_emergency': '安全紧急情况', 
            'elderly_help': '老年人求助',
            'children_safety': '儿童安全',
            'disabled_assistance': '残疾人协助'
        }

    def _load_care_phrases(self):
        """加载关怀用语 - 大幅扩充"""
        return {
            'morning': ['早上好！', '新的一天开始了，愿您有个好心情', '晨光中为您服务'],
            'afternoon': ['下午好！', '午后的时光很美好', '下午茶时间，有什么可以帮您？'],
            'evening': ['晚上好！', '辛苦了一天，请好好休息', '夜晚的宁静中我们依然守护'],
            'complaint_comfort': [
                '理解您的感受，我们会尽快解决',
                '确实让人困扰，我们正在处理',
                '我们非常重视您反映的问题',
                '请放心，我们会认真负责地处理',
                '您的满意是我们的追求',
                '这个问题我们一定会妥善解决'
            ],
            'praise_response': [
                '感谢您的认可，这是我们的荣幸',
                '您的鼓励让我们倍感温暖',
                '我们会继续保持优质服务',
                '有您的支持，我们会做得更好',
                '您的满意是我们最大的动力',
                '我们会继续努力，不辜负您的信任'
            ],
            'urgent_comfort': [
                '请保持冷静，我们正在紧急处理',
                '安全第一，我们已启动应急预案',
                '请放心，专业人员正在赶来的路上',
                '我们会全程跟进，确保问题解决',
                '紧急情况已上报，正在全力处置',
                '我们正在协调多方资源解决问题'
            ],
            'medical_advice': [
                '建议同时拨打120急救电话',
                '请保持通讯畅通，医疗团队正在行动',
                '如有紧急变化请立即联系我们',
                '我们会提供持续的医疗关注',
                '已联系社区医生，请保持电话畅通',
                '急救资源已协调，请描述详细症状'
            ],
            'safety_advice': [
                '请远离危险区域，确保自身安全',
                '专业人员正在处理，请勿靠近',
                '已设置安全警戒线，请绕行',
                '我们会确保现场安全后再进行处理',
                '安全第一，请配合现场指挥'
            ],
            'closing': [
                '祝您生活愉快！',
                '有任何问题随时联系我们',
                '感谢您对社区工作的支持',
                '我们一直在您身边',
                '愿您今天有个好心情',
                '期待再次为您服务'
            ]
        }
    
    def _load_solution_knowledge(self):
        """加载人性化的解决方案 - 大幅扩充"""
        return {
            '安全治理': {
                '高': [
                    '🚨 立即启动安全应急预案，安保人员5分钟内到场处置',
                    '🔒 联合社区民警开展紧急处置，确保人员安全撤离',
                    '📞 建立紧急通讯通道，每30分钟汇报处理进展',
                    '👮 临时封锁危险区域，设置多重安全警示标识',
                    '📱 通过社区广播和群发系统通知相关居民紧急避险',
                    '🚒 协调消防、公安等专业力量联合处置',
                    '🏥 预置医疗救援资源，做好应急救治准备'
                ],
                '中': [
                    '⏰ 2小时内安排专业人员进行现场勘查和风险评估',
                    '🛠️ 制定详细整改方案，24小时内消除安全隐患',
                    '📊 加强该区域监控密度，巡逻频次提升至每小时1次',
                    '👥 组织相关人员开展专项安全培训和演练',
                    '✅ 3日内完成整改并通过第三方安全验收',
                    '📝 建立安全隐患台账，定期回头看',
                    '🔔 设置安全提醒标识，增强居民安全意识'
                ],
                '低': [
                    '📝 纳入月度安全检查计划重点整改项',
                    '💡 结合专家和居民建议优化整体安全方案',
                    '🔍 开展全社区同类问题拉网式排查',
                    '🌟 持续改进安全防护措施和应急预案',
                    '🏆 评选安全示范户，推广先进经验',
                    '📚 开展安全知识普及活动，提升居民自救能力',
                    '🤝 建立居民安全志愿者队伍，共建安全社区'
                ]
            },
            '环境卫生': {
                '高': [
                    '🧹 紧急调派专业保洁团队，30分钟内到达现场处置',
                    '🚿 进行彻底消毒清洁，使用专业设备消除污染源',
                    '🗑️ 立即调整垃圾清运频次，今日增加2次清运',
                    '👀 设立临时监督岗，24小时监控确保清理质量',
                    '📋 今日内完成整改并通过居民验收确认',
                    '🌡️ 进行环境质量检测，确保各项指标达标',
                    '🚫 排查污染源头，采取根本性解决措施'
                ],
                '中': [
                    '🧽 今日组织专项清理，重点解决突出环境卫生问题',
                    '🔄 优化清洁流程和标准，建立长效保洁机制',
                    '🔧 全面检查维护环卫设施设备，确保正常运行',
                    '📈 2日内明显改善，一周内彻底解决问题',
                    '👃 加强通风和除味处理，改善空气质量',
                    '🌿 增加绿化美化措施，提升环境品质',
                    '📊 建立环境卫生巡查日志，每日记录整改情况'
                ],
                '低': [
                    '📅 纳入日常清洁计划重点优化项目',
                    '🌿 结合季节特点制定专项环境整治方案',
                    '👍 持续提升环境卫生精细化管理水平',
                    '💫 打造美丽宜居的社区环境',
                    '🎯 针对薄弱环节制定改进措施',
                    '🌈 开展环境卫生评比活动，激励先进',
                    '❤️ 培养居民环保意识，共建美好家园'
                ]
            },
            '民生服务': {
                '高': [
                    '⚡ 开启民生服务绿色通道，指定专人一对一服务',
                    '📞 30分钟内响应联系，2小时内提出具体解决方案',
                    '🔧 立即提供临时应急措施，最大限度缓解当前困难',
                    '👨‍💼 启动多部门协同机制，确保问题高效彻底解决',
                    '✅ 24小时内完成处理并进行满意度回访',
                    '💰 必要时启动应急资金，解决居民燃眉之急',
                    '🏠 提供临时安置或替代方案，保障基本生活需求'
                ],
                '中': [
                    '💼 今日内主动联系居民了解详细需求和困难',
                    '🔄 按照标准化服务流程提供专业解决方案',
                    '🤝 协调社区内外相关资源，提供最佳服务方案',
                    '📞 3个工作日内给予明确答复和解决方案',
                    '📋 建立服务档案，跟踪问题解决全过程',
                    '⭐ 提供多种选择方案，尊重居民意愿',
                    '🔍 深入分析问题根源，防止类似问题重复发生'
                ],
                '低': [
                    '📚 认真研究建议的可行性和实施价值',
                    '🛠️ 在服务升级和流程优化中参考采纳',
                    '🌟 持续优化服务体验和办事流程',
                    '💝 感谢居民宝贵意见，推动服务进步',
                    '💡 建立建议采纳激励机制',
                    '🔄 定期梳理居民建议，改进服务工作',
                    '🎉 对优秀建议给予表彰和奖励'
                ]
            },
            '医疗卫生': {
                '高': [
                    '🏥 立即启动紧急医疗响应程序，成立应急医疗小组',
                    '👨‍⚕️ 社区医生5分钟内联系指导，提供专业医疗建议',
                    '🚑 协助呼叫120急救，同时派人现场协助和陪伴',
                    '📱 建立紧急医疗关注档案，每30分钟跟踪一次',
                    '💊 提供必要的急救指导和药品支持',
                    '❤️ 协调医院绿色通道，确保及时救治',
                    '👥 安排志愿者或工作人员全程陪同就医'
                ],
                '中': [
                    '🩺 2小时内安排专业医疗咨询和健康评估',
                    '📋 协调优质医疗资源，提供针对性健康服务',
                    '❤️ 建立个人健康关怀档案，定期跟踪回访',
                    '👀 持续关注健康状况，提供必要医疗帮助',
                    '💡 制定个性化健康管理方案和建议',
                    '🏃 组织健康促进活动，提升健康素养',
                    '📚 提供疾病预防和健康保健知识'
                ],
                '低': [
                    '📊 纳入社区健康服务年度改善计划',
                    '🎯 丰富完善社区医疗资源配置和服务内容',
                    '📢 定期开展形式多样的健康教育活动',
                    '🌟 关爱每一位居民的身心健康',
                    '🌿 推广健康生活方式，预防疾病发生',
                    '🤝 与专业医疗机构建立长期合作关系',
                    '💝 营造关爱健康、互帮互助的社区氛围'
                ]
            },
            '文化生活': {
                '高': [
                    '🎭 立即调整活动安排，最大限度减少对居民的影响',
                    '🔄 优化活动流程和服务，确保居民参与体验',
                    '👥 加强现场组织管理，提供周到服务保障',
                    '💬 建立快速反馈机制，及时解决突发问题',
                    '✅ 今日内解决突出问题并做好解释工作',
                    '📞 设立文化活动服务热线，随时接受咨询',
                    '🎁 对受影响居民给予适当补偿和关怀'
                ],
                '中': [
                    '📝 认真研究居民的文化需求和兴趣爱好',
                    '🎨 优化活动内容和形式，提高吸引力和参与度',
                    '🛠️ 改善文化设施条件，创造良好活动环境',
                    '🌈 让社区文化生活更加丰富多彩',
                    '👥 培养文化骨干，带动居民参与',
                    '📅 制定年度文化活动计划，满足多元需求',
                    '🌟 打造社区文化品牌，提升文化品位'
                ],
                '低': [
                    '💡 居民建议已纳入社区文化建设规划',
                    '🚀 持续提升文化服务品质和内涵',
                    '❤️ 培育积极健康和谐的社区文化',
                    '🌟 文化让社区生活更加美好幸福',
                    '🎯 结合居民需求创新文化活动形式',
                    '🏆 评选文化先进，发挥示范作用',
                    '🤝 促进邻里文化交流，增进相互理解'
                ]
            },
            '交通管理': {
                '高': [
                    '🚦 立即派人现场疏导，10分钟内恢复基本交通秩序',
                    '⚠️ 设置完善的安全指引和警示标识系统',
                    '👮 协调交警部门联合处置，确保交通畅通安全',
                    '📞 建立应急通讯指挥系统，实时调度处置',
                    '✅ 2小时内解决主要问题，24小时持续优化',
                    '🚧 必要时实施临时交通管制，确保安全',
                    '📱 通过多种渠道发布交通提示信息'
                ],
                '中': [
                    '🛠️ 今日内完成相关交通设施的检修和完善',
                    '🚗 优化停车管理和交通引导方案并立即实施',
                    '🔧 全面整治影响通行的设施和环境问题',
                    '💫 让居民出行更加便捷舒适安全',
                    '📊 分析交通流量，科学规划交通组织',
                    '🚸 完善人行道和无障碍设施',
                    '🅿️ 合理设置停车位，规范停车秩序'
                ],
                '低': [
                    '📐 优化社区整体交通规划和设施布局',
                    '🅿️ 持续改善停车条件和通行环境',
                    '📚 加强交通安全宣传教育',
                    '🤝 共建安全有序文明交通环境',
                    '🌳 结合环境整治改善交通条件',
                    '💡 运用智能化手段提升交通管理水平',
                    '👥 发动居民参与交通治理，形成共治共享'
                ]
            },
            '社区治理': {
                '高': [
                    '⚖️ 立即启动矛盾纠纷应急调解机制',
                    '🤝 组织相关方面对面沟通协商解决',
                    '📞 建立24小时协调沟通专线',
                    '✅ 依法依规公平公正妥善处理',
                    '❤️ 维护和谐融洽的邻里关系',
                    '👥 邀请专业人士参与调解',
                    '📝 形成调解协议，确保执行落实'
                ],
                '中': [
                    '💬 搭建沟通平台促进相互理解和包容',
                    '📋 完善相关管理制度和服务规范',
                    '👥 加强社区服务协调和资源整合',
                    '🏡 共建美好和谐幸福家园',
                    '🌟 推广邻里互助的好经验好做法',
                    '🎯 针对性地开展社区关系建设',
                    '🤲 培养居民参与社区治理的意识和能力'
                ],
                '低': [
                    '📝 认真记录居民宝贵意见建议',
                    '🔄 在社区建设和发展中充分参考采纳',
                    '🌟 促进邻里和睦相处互帮互助',
                    '💝 感谢居民为社区建设出谋划策',
                    '💡 建立居民建议采纳应用机制',
                    '🏆 表彰社区建设贡献突出的居民',
                    '🌱 培育社区共同体意识'
                ]
            }
        }
    
    def _load_response_templates(self):
        """加载温暖人心的回复模板 - 大幅扩充"""
        return {
            '抱怨': {
                '高': {
                    'template': "🚨 {time_greeting}听到您反映的情况，我们高度重视！{immediate_action}。{comfort}，{department}正在紧急处理，{sla}内会给您明确进展。{solution}",
                    'immediate_action': '已启动应急预案，成立专项处置小组',
                    'comfort': '理解您的紧急需求和担忧'
                },
                '中': {
                    'template': "⚠️ {time_greeting}感谢您告诉我们这个问题！{comfort}，{department}已接手处理，{sla}内会向您反馈进展。{solution}",
                    'immediate_action': '已安排专人负责，制定解决方案',
                    'comfort': '我们理解这给您带来了不便和困扰'
                },
                '低': {
                    'template': "✅ {time_greeting}谢谢您的反馈！{comfort}，我们已记录并会认真研究。{department}将在{sla}内给您回复。{solution}",
                    'immediate_action': '已纳入改善计划，组织专题研究',
                    'comfort': '您的建议对我们改进工作很重要'
                }
            },
            '表扬': {
                '高': {
                    'template': "🌟 {time_greeting}看到您的表扬，我们整个团队都备受鼓舞！{appreciation}，{department}会珍惜这份认可，{continue_effort}。{solution}",
                    'appreciation': '衷心感谢您的肯定和鼓励',
                    'continue_effort': '继续努力提供更优质的服务'
                },
                '中': {
                    'template': "👍 {time_greeting}非常感谢您的鼓励！{appreciation}，{department}会以此为契机，{improvement}。{solution}",
                    'appreciation': '您的认可让我们倍感温暖和振奋',
                    'improvement': '进一步提升服务水平和质量'
                },
                '低': {
                    'template': "😊 {time_greeting}谢谢您的支持！{appreciation}，{department}会继续保持，{commitment}。{solution}",
                    'appreciation': '很高兴能为您提供满意的服务',
                    'commitment': '努力做得更好，不辜负您的信任'
                }
            },
            '求助': {
                '高': {
                    'template': "🆘 {time_greeting}检测到您的紧急求助！{immediate_action}。{comfort}，{department}正在全力协助，{sla}内会主动联系您。{solution}",
                    'immediate_action': '已启动紧急响应程序，协调多方资源',
                    'comfort': '请保持冷静，专业人员正在赶来的路上'
                },
                '中': {
                    'template': "ℹ️ {time_greeting}收到您的求助信息！{comfort}，{department}会尽快提供帮助，{sla}内给您回复。{solution}",
                    'immediate_action': '已安排专业团队提供协助支持',
                    'comfort': '我们理解您需要帮助，请放心'
                },
                '低': {
                    'template': "💡 {time_greeting}了解您的需求！{comfort}，{department}会研究如何更好帮助您，{sla}内反馈。{solution}",
                    'immediate_action': '已记录支持需求，制定帮助方案',
                    'comfort': '我们会尽力为您提供所需的帮助'
                }
            }
        }
    
    def _load_workflow_rules(self):
        """加载人性化的工作流 - 优化版"""
        return {
            '高': {
                'department': '紧急响应组',
                'sla': '立即响应',
                'priority': 1,
                'response_time': '5分钟内响应',
                'follow_up': '每小时更新进展',
                'color': '#ff4444',
                'icon': '🚨'
            },
            '中': {
                'department': '专业处理组', 
                'sla': '24小时',
                'priority': 2,
                'response_time': '2小时内响应',
                'follow_up': '每日汇报进度',
                'color': '#ffaa00',
                'icon': '⚠️'
            },
            '低': {
                'department': '服务改善组',
                'sla': '3个工作日',
                'priority': 3,
                'response_time': '当天内响应',
                'follow_up': '处理完成时反馈',
                'color': '#00aa00',
                'icon': '✅'
            }
        }
    
    def _load_keyword_triggers(self):
        """加载情感关键词 - 大幅扩充"""
        return {
            'safety_keywords': ['火灾', '漏电', '坠物', '被困', '急救', '危险', '隐患', '安全', '保安', '消防', '触电', '燃气泄漏', '爆炸', '坍塌', '高空抛物'],
            'medical_keywords': ['头晕', '呕吐', '胸痛', '呼吸困难', '昏迷', '急救', '发病', '摔伤', '高血压', '糖尿病', '心脏病', '中风', '发烧', '骨折', '流血'],
            'urgency_keywords': ['立即', '马上', '紧急', '赶快', '立刻', '速来', '救命', '赶紧', '刻不容缓', '火速', '即刻', '火急'],
            'help_keywords': ['求助', '帮忙', '协助', '帮帮我', '需要帮助', '不知道怎么', '能否帮助', '请求支援', '急需帮助', '求援'],
            'elderly_keywords': ['老人', '老年人', '大爷', '大妈', '爷爷', '奶奶', '老人家', '高龄', '退休'],
            'children_keywords': ['小孩', '儿童', '孩子', '小朋友', '婴儿', '宝宝', '学生', '幼儿'],
            'disabled_keywords': ['残疾', '残疾人', '行动不便', '轮椅', '盲人', '聋哑'],
            'emotion_keywords': {
                'angry': ['愤怒', '气愤', '恼火', '受不了', '太过分', '岂有此理', '生气', '火大', '愤怒', '暴怒', '气愤填膺'],
                'anxious': ['担心', '焦虑', '着急', '不安', '困扰', '烦恼', '忧虑', '心急', '焦躁', '坐立不安'],
                'frustrated': ['失望', '无奈', '沮丧', '心力交瘁', '疲惫', '累觉不爱', '绝望', '灰心', '泄气'],
                'worried': ['担忧', '揪心', '睡不好', '吃不下', '担心', '忧愁', '发愁', '提心吊胆']
            },
            'location_keywords': ['楼道', '电梯', '车库', '花园', '大门', '屋顶', '地下室', '小区', '停车场', '消防通道', '活动室', '健身房', '游泳池'],
            'positive_keywords': ['感谢', '谢谢', '点赞', '表扬', '好评', '满意', '棒', '好', '优秀', '赞', '贴心', '周到', '专业']
        }
    
    def _load_special_scenarios(self):
        """加载特殊场景处理规则 - 大幅扩充"""
        return {
            'medical_emergency': {
                'keywords': ['头晕呕吐', '胸痛', '昏迷', '呼吸困难', '心脏病', '中风', '大出血', '窒息', '休克'],
                'auto_upgrade_urgency': True,
                'special_advice': 'medical_advice',
                'priority_boost': 2
            },
            'safety_emergency': {
                'keywords': ['火灾', '漏电', '被困', '燃气泄漏', '爆炸', '坍塌', '毒气'],
                'auto_upgrade_urgency': True,
                'special_advice': 'safety_advice',
                'priority_boost': 2
            },
            'elderly_help': {
                'keywords': ['老人', '老年人', '大爷', '大妈', '爷爷', '奶奶', '老人家', '高龄'],
                'special_attention': True,
                'priority_boost': 1,
                'care_type': 'elderly_care'
            },
            'children_safety': {
                'keywords': ['小孩', '儿童', '孩子', '小朋友', '婴儿', '宝宝', '学生'],
                'special_attention': True,
                'priority_boost': 1,
                'care_type': 'children_care'
            },
            'disabled_assistance': {
                'keywords': ['残疾', '残疾人', '行动不便', '轮椅', '盲人', '聋哑'],
                'special_attention': True,
                'priority_boost': 1,
                'care_type': 'disabled_care'
            }
        }
    
    def _load_seasonal_knowledge(self):
        """加载季节性知识 - 新增"""
        return {
            'summer': {
                '空调问题': '夏季空调使用高峰，维修响应时间缩短50%',
                '蚊虫治理': '增加公共区域消杀频次，提供防蚊指导',
                '用电安全': '加强用电设备检查，防止过载引发火灾',
                '防暑降温': '开放避暑场所，提供防暑药品'
            },
            'winter': {
                '供暖保障': '提前检查供暖设备，确保正常供暖',
                '防冻措施': '加强水管防冻保护，准备应急抢修',
                '冰雪清理': '储备除雪物资，及时清理积雪',
                '老人关怀': '加强独居老人走访，提供保暖帮助'
            },
            'rainy_season': {
                '防水防漏': '检查房屋渗漏，及时维修',
                '防滑措施': '设置防滑警示，清理积水',
                '排水疏通': '清理排水系统，防止内涝',
                '电器安全': '检查室外电器，防止漏电'
            }
        }
    
    def detect_special_scenario(self, text, prediction):
        """检测特殊场景 - 增强版"""
        text_lower = text.lower()
        detected_scenarios = []
        
        for scenario_name, rules in self.special_scenarios.items():
            if any(keyword in text_lower for keyword in rules['keywords']):
                detected_scenarios.append(scenario_name)
                
                # 自动升级紧急程度
                if rules.get('auto_upgrade_urgency', False) and prediction['urgency'] != '高':
                    prediction['urgency'] = '高'
                    
        return detected_scenarios
    
    def get_seasonal_advice(self, category):
        """获取季节性建议 - 新增方法"""
        from datetime import datetime
        month = datetime.now().month
        
        if 6 <= month <= 8:  # 夏季
            return self.seasonal_knowledge['summer'].get(category, '')
        elif 12 <= month or month <= 2:  # 冬季
            return self.seasonal_knowledge['winter'].get(category, '')
        elif 3 <= month <= 5 or 9 <= month <= 11:  # 雨季
            return self.seasonal_knowledge['rainy_season'].get(category, '')
        return ''
    
    def get_solutions(self, category, urgency, count=3):
        """获取解决方案"""
        solutions = self.solution_knowledge.get(category, {}).get(urgency, [])
        return solutions[:count]
    
    def get_response_template(self, emotion, urgency):
        """获取回复模板"""
        return self.response_templates.get(emotion, {}).get(urgency, {})
    
    def get_workflow(self, urgency):
        """获取工作流配置"""
        return self.workflow_rules.get(urgency, {})
    
    def get_care_phrase(self, phrase_type):
        """获取关怀用语"""
        phrases = self.care_phrases.get(phrase_type, [])
        import random
        return random.choice(phrases) if phrases else ""
    
    def get_time_greeting(self):
        """获取时间问候语"""
        from datetime import datetime
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return self.get_care_phrase('morning')
        elif 12 <= hour < 18:
            return self.get_care_phrase('afternoon')
        else:
            return self.get_care_phrase('evening')
    
    def generate_auto_response(self, prediction, text="", custom_data=None):
        """生成温暖人心的自动回复 - 紧急修复版"""
        emotion = prediction['emotion']
        urgency = prediction['urgency']
        category = prediction['category']
        
        # 紧急修复：医疗求助场景情感重映射
        text_lower = text.lower()
        medical_emergency_keywords = ['头晕', '呕吐', '胸痛', '昏迷', '呼吸困难', '急救', '发病']
        
        if (category == '医疗卫生' and 
            any(keyword in text_lower for keyword in medical_emergency_keywords) and
            any(word in text_lower for word in ['求助', '帮忙', '帮助', '急需'])):
            emotion = '求助'  # 强制重映射为求助
        
        # 检测特殊场景
        special_scenarios = self.detect_special_scenario(text, prediction)
        
        template_data = self.get_response_template(emotion, urgency)
        workflow_data = self.get_workflow(urgency)
        solutions = self.get_solutions(category, urgency, 2)
        
        if not template_data:
            return "感谢您的反馈！我们已收到您的信息，会尽快处理。"
        
        # 构建回复数据
        response_data = {
            'department': workflow_data.get('department', '相关团队'),
            'sla': workflow_data.get('sla', '尽快'),
            'immediate_action': template_data.get('immediate_action', '已安排处理'),
            'comfort': template_data.get('comfort', '理解您的感受'),
            'appreciation': template_data.get('appreciation', '感谢您的反馈'),
            'continue_effort': template_data.get('continue_effort', '继续努力服务'),
            'improvement': template_data.get('improvement', '不断改进提升'),
            'commitment': template_data.get('commitment', '用心服务'),
            'solution': " ".join(solutions[:2]),
            'time_greeting': self.get_time_greeting()
        }
        
        # 特殊场景处理
        additional_advice = ""
        if 'medical_emergency' in special_scenarios:
            response_data['comfort'] = self.get_care_phrase('urgent_comfort')
            additional_advice = self.get_care_phrase('medical_advice')
        elif 'safety_emergency' in special_scenarios:
            response_data['comfort'] = self.get_care_phrase('urgent_comfort')
            additional_advice = self.get_care_phrase('safety_advice')
        
        # 添加季节性建议
        seasonal_advice = self.get_seasonal_advice(category)
        if seasonal_advice:
            additional_advice += " " + seasonal_advice
        
        # 合并自定义数据
        if custom_data:
            response_data.update(custom_data)
        
        # 渲染模板
        template = template_data.get('template', '感谢您的反馈，我们会及时处理。')
        try:
            response = template.format(**response_data)
        except KeyError as e:
            print(f"模板渲染错误: {e}")
            response = template
        
        # 添加额外建议
        if additional_advice:
            response += " " + additional_advice
        
        # 添加关怀结语
        if emotion == '抱怨':
            response += " " + self.get_care_phrase('complaint_comfort')
        elif emotion == '求助':
            response += " " + self.get_care_phrase('urgent_comfort')
        else:
            response += " " + self.get_care_phrase('praise_response')
        
        return response
    
    def get_processing_timeline(self, urgency):
        """获取人性化的处理时间线 - 增强版"""
        timelines = {
            '高': [
                {'time': '立即', 'action': '接单并紧急联系', 'icon': '🚨'},
                {'time': '30分钟内', 'action': '专业人员到场处理', 'icon': '👨‍💼'},
                {'time': '2小时内', 'action': '初步控制局面', 'icon': '✅'},
                {'time': '24小时内', 'action': '彻底解决并回访', 'icon': '📞'}
            ],
            '中': [
                {'time': '2小时内', 'action': '接单并了解情况', 'icon': '📱'},
                {'time': '24小时内', 'action': '现场勘查制定方案', 'icon': '🔍'},
                {'time': '3个工作日内', 'action': '完成处理并确认', 'icon': '✅'}
            ],
            '低': [
                {'time': '当天内', 'action': '登记并安排处理', 'icon': '📝'},
                {'time': '3个工作日内', 'action': '研究制定改进方案', 'icon': '💡'},
                {'time': '1周内', 'action': '落实改进并反馈', 'icon': '🌟'}
            ]
        }
        return timelines.get(urgency, [])
    
    def analyze_text_keywords(self, text):
        """分析文本情感和关键词 - 场景名称中文化"""
        analysis = {
            'safety_related': any(keyword in text for keyword in self.keyword_triggers['safety_keywords']),
            'medical_related': any(keyword in text for keyword in self.keyword_triggers['medical_keywords']),
            'high_urgency': any(keyword in text for keyword in self.keyword_triggers['urgency_keywords']),
            'need_help': any(keyword in text for keyword in self.keyword_triggers['help_keywords']),
            'involves_elderly': any(keyword in text for keyword in self.keyword_triggers['elderly_keywords']),
            'involves_children': any(keyword in text for keyword in self.keyword_triggers['children_keywords']),
            'involves_disabled': any(keyword in text for keyword in self.keyword_triggers['disabled_keywords']),
            'emotional_state': [],
            'locations': [],
            'is_positive': any(keyword in text for keyword in self.keyword_triggers['positive_keywords']),
            'special_scenarios': []
        }
        
        # 检测情绪关键词
        for emotion_type, keywords in self.keyword_triggers['emotion_keywords'].items():
            if any(keyword in text for keyword in keywords):
                analysis['emotional_state'].append(emotion_type)
        
        # 检测位置信息
        analysis['locations'] = [loc for loc in self.keyword_triggers['location_keywords'] if loc in text]
        
        # 检测特殊场景并转换为中文名称
        text_lower = text.lower()
        for scenario_name, rules in self.special_scenarios.items():
            if any(keyword in text_lower for keyword in rules['keywords']):
                # 将英文场景名称转换为中文
                chinese_name = self.scenario_name_mapping.get(scenario_name, scenario_name)
                analysis['special_scenarios'].append(chinese_name)
        
        return analysis
    
    def get_category_description(self, category):
        """获取类别描述"""
        descriptions = {
            '安全治理': '守护您和家人的生命财产安全',
            '环境卫生': '共创整洁美丽舒适的居住环境',
            '民生服务': '提供温暖贴心便捷的生活服务',
            '医疗卫生': '关爱每一位居民的身心健康',
            '文化生活': '丰富社区居民的精神文化生活',
            '交通管理': '保障安全便捷有序的出行环境',
            '社区治理': '共建和谐美好幸福的社区家园'
        }
        return descriptions.get(category, '社区生活相关服务')

# 单例知识库实例
knowledge_base = CommunityKnowledgeBase()