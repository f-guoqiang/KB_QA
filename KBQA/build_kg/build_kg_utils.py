#encoding:utf8
import os
import re
import json
import codecs
import threading
from py2neo import Graph
import pandas as pd 
import numpy as np 
from tqdm import tqdm 

#传入的是medical.json文件,里面有疾病的详细描述和治疗方法，通过这个文件创建实体和关系
'''{"_id" : { "$oid" : "5bb578b6831b973a137e3eed" }, 
"name" : "大叶性肺炎", 
"desc" : "大叶性肺炎(lobarpneumonia)，又名肺炎球菌肺炎，是由肺炎双球菌等细菌感染引起的呈大叶性分布的急性肺实质炎症。好发于青壮年男性和冬春季节，常见诱因有受凉、淋雨、醉酒或全身麻醉手术后、镇静剂过量等，当这些诱因使呼吸道防御功能被削弱时，细菌侵入肺泡通过变态反应使肺泡壁毛细血管通透性增强，浆液及纤维素渗出，富含蛋白的渗出物中细菌迅速繁殖，并向邻近肺组织蔓延，波及一个肺段或整个肺叶。近年来由于大量强有力抗生素的使用，典型的大叶性肺炎已较少见到。临床症状有突然寒战、高热、胸痛、咳嗽、咳铁锈色痰，部分患者有恶心、呕吐及烦躁不安、谵妄等消化系统和神经系统症状。体征有急性病容，呼吸急促，鼻翼扇动，早期肺部体征不明显或仅有呼吸音减低和胸膜摩擦音;实变期可有典型体征，如患侧呼吸运动减弱，语颤增强，叩诊浊音，听诊呼吸音减低，有湿罗音或病理性支气管呼吸音。血白细胞计数及中性粒细胞增高;典型的X线表现为肺段、叶实变。该病病程短，及时应用青霉素等抗生素治疗可痊愈。", 
"category" : [ "疾病百科", "内科", "呼吸内科" ], 
"prevent" : "1、注意预防上呼吸道感染，加强耐寒锻炼。\n2、避免淋雨受寒，醉酒，过劳等诱因。\n3、积极治疗原发病，如慢性心肺疾病，慢性肝炎，糖尿病和口腔疾病等，可以预防大叶性肺炎。", 
"cause" : "多种细菌均可引起大叶性肺炎，但绝大多数为肺炎链球菌。肺炎链球菌为革兰阳性球菌，有荚膜，其致病力是由于高分子多糖体的荚膜对组织的侵袭作用。少数为肺炎杆菌、金黄色葡萄球菌、溶血性链球菌、流感嗜血杆菌等。\n当机体受寒、过度疲劳、醉酒、感冒、糖尿病、免疫功能低下等使呼吸道防御功能被削弱，细菌侵入肺泡通过变态反应使肺泡壁毛细血管通透性增强，浆液及纤维素渗出，富含蛋白的渗出物中细菌迅速繁殖，并通过肺泡间孔或呼吸细支气管向邻近肺组织蔓延，波及一个肺段或整个肺叶。大叶间的蔓延系带菌的渗出液经叶支气管播散所致。\n大叶肺炎病变起始于局部肺泡，并迅速蔓延至一个肺段或整个大叶。临床上起病急骤，病程大约一周。常以高热、恶寒开始，继而出现胸痛、咳嗽、咳铁锈色痰、呼吸困难、并有肺实变体征及外周血白细胞计数增高等。", "symptom" : [ "湿啰音", "胸痛", "发烧", "咳铁锈色痰", "急性面容", "呼吸音减弱" ], "yibao_status" : "否", 
"get_prob" : "0.4%", 
"easy_get" : "多发生于青壮年男性", "get_way" : "无传染性", "acompany" : [ "脓胸" ], "cure_department" : [ "内科", "呼吸内科" ], "cure_way" : [ "青霉素等抗生素药物治疗", "对症支持性治疗", "并发症治疗" ], "cure_lasttime" : "7--10天", "cured_prob" : "90%以上", "common_drug" : [ "乳酸左氧氟沙星片", "阿奇霉素片" ], "cost_money" : "根据不同医院，收费标准不一致，省市三甲医院约（3000-8000元）", "check" : [ "Optochin敏感试验", "小白鼠毒力试验", "痰培养", "肺活量体重指数", "胸部平片", "免疫电泳", "血常规", "痰液细菌涂片检查" ], "do_eat" : [ "栗子（熟）", "鲫鱼", "猪肉(瘦)", "油菜" ], "not_eat" : [ "洋葱", "辣椒(青、尖)", "辣椒(红、尖、干)", "韭菜" ], "recommand_eat" : [ "奶汤锅子鱼", "酱豆腐汁烧猪肉", "百合汤", "山楂百合汤", "番茄鸡蛋煎饼", "凉拌番茄", "番茄猪肝瘦肉汤", "番茄牛肉" ], "recommand_drug" : [ "阿奇霉素胶囊", "阿奇霉素分散片", "诺氟沙星胶囊", "阿奇霉素片", "乳酸左氧氟沙星片" ],
"drug_detail" : [ "宜昌长江乳酸左氧氟沙星片(乳酸左氧氟沙星片)", "希舒美(阿奇霉素片)", "维宏(阿奇霉素片)", "宜昌长江阿奇霉素分散片(阿奇霉素分散片)", "大连天宇制药阿奇霉素胶囊(阿奇霉素胶囊)", "兰花药业诺氟沙星胶囊(诺氟沙星胶囊)", "福邦药业阿奇霉素胶囊(阿奇霉素胶囊)", "葵花药业得菲尔公司阿奇霉素(阿奇霉素胶囊)", "利民制药阿奇霉素胶囊(阿奇霉素胶囊)", "东药阿奇霉素胶囊(阿奇霉素胶囊)", "浙江南洋药业阿奇霉素胶囊(阿奇霉素胶囊)", "江苏长江阿奇霉素胶囊(阿奇霉素胶囊)", "辅仁药业阿奇霉素片(阿奇霉素片)", "汇仁药业阿奇霉素片(阿奇霉素片)", "爱普森药业阿奇霉素片(阿奇霉素片)", "石药欧意诺氟沙星胶囊(诺氟沙星胶囊)", "白云山医药诺氟沙星胶囊(诺氟沙星胶囊)", "康美药业诺氟沙星胶囊(诺氟沙星胶囊)", "锦华药业诺氟沙星胶囊(诺氟沙星胶囊)", "北京京丰制药集团诺氟沙星胶(诺氟沙星胶囊)", "欣匹特(阿奇霉素分散片)", "海王阿奇霉素片(阿奇霉素片)", "安诺药业阿奇霉素分散片(阿奇霉素分散片)" ] }'''


#从Jason数据库中提取三元组，传入neo4j中

#这个函数
def print_data_info(data_path): 
    
    triples = []
    i = 0
    with open(data_path,'r',encoding='utf8') as f:
        for line in f.readlines():
            data = json.loads(line)
            
            print(json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '),ensure_ascii=False))
            i += 1
            if i >=5:
                break
    return triples

class MedicalExtractor(object):
    def __init__(self):
        super(MedicalExtractor, self).__init__()
        self.graph = Graph(
            host="127.0.0.1",
            http_port=7474,
            user="neo4j",
            password="1")

        # 共8类节点
        self.drugs = [] # 药品
        self.recipes = [] #菜谱
        self.foods = [] #　食物
        self.checks = [] # 检查
        self.departments = [] #科室
        self.producers = [] #药企
        self.diseases = [] #疾病
        self.symptoms = []#症状

        self.disease_infos = []#疾病信息 ，疾病实体是有属性的，其他实体没有

        # 构建节点实体关系
        self.rels_department = [] #　科室－科室关系
        self.rels_noteat = [] # 疾病－忌吃食物关系
        self.rels_doeat = [] # 疾病－宜吃食物关系
        self.rels_recommandeat = [] # 疾病－推荐吃食物关系
        self.rels_commonddrug = [] # 疾病－通用药品关系
        self.rels_recommanddrug = [] # 疾病－热门药品关系
        self.rels_check = [] # 疾病－检查关系
        self.rels_drug_producer = [] # 厂商－药物关系

        self.rels_symptom = [] #疾病症状关系
        self.rels_acompany = [] # 疾病并发关系
        self.rels_category = [] #　疾病与科室之间的关系
        
    def extract_triples(self,data_path):
        print("从json文件中转换抽取三元组")
        with open(data_path,'r',encoding='utf8') as f:
            for line in tqdm(f.readlines(),ncols=80):
                data_json = json.loads(line)
                disease_dict = {}
                disease = data_json['name']
                disease_dict['name'] = disease
                self.diseases.append(disease)
                disease_dict['desc'] = ''
                disease_dict['prevent'] = ''
                disease_dict['cause'] = ''
                disease_dict['easy_get'] = ''
                disease_dict['cure_department'] = ''
                disease_dict['cure_way'] = ''
                disease_dict['cure_lasttime'] = ''
                disease_dict['symptom'] = ''
                disease_dict['cured_prob'] = ''

                if 'symptom' in data_json:
                    self.symptoms += data_json['symptom']
                    
                    for symptom in data_json['symptom']:#症状是序列
                        self.rels_symptom.append([disease,'has_symptom', symptom])

                if 'acompany' in data_json:
                    for acompany in data_json['acompany']:
                        self.rels_acompany.append([disease,'acompany_with', acompany])
                        self.diseases.append(acompany)

                if 'desc' in data_json:
                    disease_dict['desc'] = data_json['desc']

                if 'prevent' in data_json:
                    disease_dict['prevent'] = data_json['prevent']

                if 'cause' in data_json:
                    disease_dict['cause'] = data_json['cause']

                if 'get_prob' in data_json:
                    disease_dict['get_prob'] = data_json['get_prob']

                if 'easy_get' in data_json:
                    disease_dict['easy_get'] = data_json['easy_get']

                if 'cure_department' in data_json:
                    cure_department = data_json['cure_department']
                    if len(cure_department) == 1:
                         self.rels_category.append([disease, 'cure_department',cure_department[0]])
                    if len(cure_department) == 2:
                        big = cure_department[0]
                        small = cure_department[1]
                        self.rels_department.append([small,'belongs_to', big])
                        self.rels_category.append([disease,'cure_department', small])

                    disease_dict['cure_department'] = cure_department
                    self.departments += cure_department

                if 'cure_way' in data_json:
                    disease_dict['cure_way'] = data_json['cure_way']

                if  'cure_lasttime' in data_json:
                    disease_dict['cure_lasttime'] = data_json['cure_lasttime']

                if 'cured_prob' in data_json:
                    disease_dict['cured_prob'] = data_json['cured_prob']

                if 'common_drug' in data_json:
                    common_drug = data_json['common_drug']
                    for drug in common_drug:
                        self.rels_commonddrug.append([disease,'has_common_drug', drug])
                    self.drugs += common_drug

                if 'recommand_drug' in data_json:
                    recommand_drug = data_json['recommand_drug']
                    self.drugs += recommand_drug
                    for drug in recommand_drug:
                        self.rels_recommanddrug.append([disease,'recommand_drug', drug])

                if 'not_eat' in data_json:
                    not_eat = data_json['not_eat']
                    for _not in not_eat:
                        self.rels_noteat.append([disease,'not_eat', _not])

                    self.foods += not_eat
                    do_eat = data_json['do_eat']
                    for _do in do_eat:
                        self.rels_doeat.append([disease,'do_eat', _do])

                    self.foods += do_eat

                if 'recommand_eat' in data_json:
                    recommand_eat = data_json['recommand_eat']
                    for _recommand in recommand_eat:
                        self.rels_recommandeat.append([disease,'recommand_recipes', _recommand])
                    self.recipes += recommand_eat

                if 'check' in data_json:
                    check = data_json['check']
                    for _check in check:
                        self.rels_check.append([disease, 'need_check', _check])
                    self.checks += check

                if 'drug_detail' in data_json:
                    for det in data_json['drug_detail']:
                        
                        #按照左括号切分
                        '''"drug_detail" : [ "宜昌长江乳酸左氧氟沙星片(乳酸左氧氟沙星片)", 
                        "希舒美(阿奇霉素片)", "维宏(阿奇霉素片)", "宜昌长江阿奇霉素分散片(阿奇霉素分散片)","'''
                        det_spilt = det.split('(')
                        if len(det_spilt) == 2:
                            #p是药企，d是药企
                            p,d = det_spilt
                            d = d.rstrip(')')
                            if p.find(d) > 0:
                                p = p.rstrip(d)
                            self.producers.append(p)
                            self.drugs.append(d)
                            self.rels_drug_producer.append([p,'production',d])
                        else:
                            d = det_spilt[0]
                            self.drugs.append(d)

                self.disease_infos.append(disease_dict)

    def write_nodes(self,entitys,entity_type):
        #entitys传入的是定义好的实体列表，entity_type是中文介绍
        print("写入 {0} 实体".format(entity_type))
        for node in tqdm(set(entitys),ncols=80):
            
            cql = """MERGE(n:{label}{{name:'{entity_name}'}})""".format(
                label=entity_type,entity_name=node.replace("'",""))
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)
        
    def write_edges(self,triples,head_type,tail_type):
        print("写入 {0} 关系".format(triples[0][1]))
        for head,relation,tail in tqdm(triples,ncols=80):
            #用merge不会多次写入
            cql = """MATCH(p:{head_type}),(q:{tail_type})
                    WHERE p.name='{head}' AND q.name='{tail}'
                    MERGE (p)-[r:{relation}]->(q)""".format(
                        head_type=head_type,tail_type=tail_type,head=head.replace("'",""),
                        tail=tail.replace("'",""),relation=relation)
            try:
                self.graph.run(cql)
            except Exception as e:
                print(e)
                print(cql)

    def set_attributes(self,entity_infos,etype):
        print("写入 {0} 实体的属性".format(etype))
        
        for e_dict in tqdm(entity_infos[892:],ncols=80):
            name = e_dict['name']
            del e_dict['name']
            for k,v in e_dict.items():
                if k in ['cure_department','cure_way']:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}={v}""".format(label=etype,name=name.replace("'",""),k=k,v=v)
                else:
                    cql = """MATCH (n:{label})
                        WHERE n.name='{name}'
                        set n.{k}='{v}'""".format(label=etype,name=name.replace("'",""),k=k,v=v.replace("'","").replace("\n",""))
                try:
                    self.graph.run(cql)
                except Exception as e:
                    print(e)
                    print(cql)


    def create_entitys(self):
        self.write_nodes(self.drugs,'药品')
        self.write_nodes(self.recipes,'菜谱')
        self.write_nodes(self.foods,'食物')
        self.write_nodes(self.checks,'检查')
        self.write_nodes(self.departments,'科室')
        self.write_nodes(self.producers,'药企')
        self.write_nodes(self.diseases,'疾病')
        self.write_nodes(self.symptoms,'症状')

    def create_relations(self):
        self.write_edges(self.rels_department,'科室','科室')
        self.write_edges(self.rels_noteat,'疾病','食物')
        self.write_edges(self.rels_doeat,'疾病','食物')
        self.write_edges(self.rels_recommandeat,'疾病','菜谱')
        self.write_edges(self.rels_commonddrug,'疾病','药品')
        self.write_edges(self.rels_recommanddrug,'疾病','药品')
        self.write_edges(self.rels_check,'疾病','检查')
        self.write_edges(self.rels_drug_producer,'药企','药品')
        self.write_edges(self.rels_symptom,'疾病','症状')
        self.write_edges(self.rels_acompany,'疾病','疾病')
        self.write_edges(self.rels_category,'疾病','科室')

    def set_diseases_attributes(self): 
        # self.set_attributes(self.disease_infos,"疾病")
        t=threading.Thread(target=self.set_attributes,args=(self.disease_infos,"疾病"))
        t.setDaemon(False)
        t.start()


    def export_data(self,data,path):
        if isinstance(data[0],str):
            data = sorted([d.strip("...") for d in set(data)])
        with codecs.open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def export_entitys_relations(self):
        self.export_data(self.drugs,'./graph_data/drugs.json')
        self.export_data(self.recipes,'./graph_data/recipes.json')
        self.export_data(self.foods,'./graph_data/foods.json')
        self.export_data(self.checks,'./graph_data/checks.json')
        self.export_data(self.departments,'./graph_data/departments.json')
        self.export_data(self.producers,'./graph_data/producers.json')
        self.export_data(self.diseases,'./graph_data/diseases.json')
        self.export_data(self.symptoms,'./graph_data/symptoms.json')

        self.export_data(self.rels_department,'./graph_data/rels_department.json')
        self.export_data(self.rels_noteat,'./graph_data/rels_noteat.json')
        self.export_data(self.rels_doeat,'./graph_data/rels_doeat.json')
        self.export_data(self.rels_recommandeat,'./graph_data/rels_recommandeat.json')
        self.export_data(self.rels_commonddrug,'./graph_data/rels_commonddrug.json')
        self.export_data(self.rels_recommanddrug,'./graph_data/rels_recommanddrug.json')
        self.export_data(self.rels_check,'./graph_data/rels_check.json')
        self.export_data(self.rels_drug_producer,'./graph_data/rels_drug_producer.json')
        self.export_data(self.rels_symptom,'./graph_data/rels_symptom.json')
        self.export_data(self.rels_acompany,'./graph_data/rels_acompany.json')
        self.export_data(self.rels_category,'./graph_data/rels_category.json')





if __name__ == '__main__':
    path = "./medical.json"
    # print_data_info(path)
    extractor = MedicalExtractor()
    extractor.extract_triples(path)
    extractor.create_entitys()
    extractor.create_relations()
    extractor.set_diseases_attributes()
    extractor.export_entitys_relations()
