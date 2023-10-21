import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from Bio import Entrez, Medline
from joblib import Parallel, delayed



device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')

NUM_CLASSES_SUPERCLASSES = 5
NUM_CLASSES = 57
BATCH_SIZE = 128

class BaselineModelSuperClasses(nn.Module):

    def __init__(self, matrix_size=30):
        super(BaselineModelSuperClasses, self).__init__()
        self.linear1 = nn.Linear(matrix_size**2, (matrix_size**2)//2)
        self.linear2 = nn.Linear((matrix_size**2)//2, matrix_size**2//4)
        self.linear3 = nn.Linear((matrix_size**2)//4, NUM_CLASSES_SUPERCLASSES)

    def forward(self, x):

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        return x
    
class BaselineModel(nn.Module):

    def __init__(self, matrix_size=30):
        super(BaselineModel, self).__init__()
        self.linear1 = nn.Linear(matrix_size**2, (matrix_size**2)//2)
        self.linear2 = nn.Linear((matrix_size**2)//2, matrix_size**2//4)
        self.linear3 = nn.Linear((matrix_size**2)//4, NUM_CLASSES)

    def forward(self, x):

        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        return x
def compute_accuracy(pred, target):
  return target.detach().numpy(), pred.argmax(-1).detach().numpy()



def get_matrix(subj,obj): 
  Entrez.email = 'noname@email.com'
  # Formulating the PubMed Query
  query = '("' + subj + '/*"[Majr:NoExp]) AND "' + obj + '/*"[Majr:NoExp]'
  # query = '("' + subj + '"[Mesh]) AND "' + obj + '"[Mesh]'
#   print(query)
  matrix = []
  # Searching for the PubMed records having the Subject and Object as MeSH Terms
  handle = Entrez.esearch(db="pubmed", retmax=20, term=query, sort="relevance")
  records1 = Entrez.read(handle)
  handle.close()
  global PubIds
  PubIds= records1["IdList"]
  NPub = len(PubIds)
  if(NPub==0):
    return matrix
  Associations = []
  for publication in PubIds:
    handle = Entrez.efetch(db="pubmed", id=publication, rettype="medline", retmode="text")
    records2 = Medline.parse(handle)
    for r in records2:
      MeSHTerms = r.get("MH", "?")
    # Extracting the Subject-Object Qualifiers Associations in MeSH Terms
    Subjs = [item.replace("*", "") for item in MeSHTerms if (item.find(subj) >= 0) and (item.find("/") >= 0)]
    Objs = [item.replace("*", "") for item in MeSHTerms if (item.find(obj) >= 0) and (item.find("/") >= 0)]
    for subject1 in Subjs:
      for object1 in Objs:
        Associations.append((subject1[subject1.find("/") + 1:], object1[object1.find("/") + 1:]))
    # Adding associations to files
    # print(Associations)
    handle.close()
  couples = []
  qualifiers = ["analysis", "pathology", "diagnosis", "etiology", "complications", "metabolism", "biosynthesis", "therapy", "drug therapy",
                "methods", "chemistry", "diagnostic imaging", "genetics", "physiopathology", "epidemiology", "blood", "pharmacology",
                "analogs & derivatives", "adverse effects", "therapeutic use", "surgery", "pharmacokinetics",
                "agonists", "ethnology", "administration & dosage", "drug effects", "enzymology", "physiology", "toxicity", "immunology"]
  for associationcouple in Associations:
    if associationcouple[0].find("/") >= 0:
        subject01 = associationcouple[0].split("/")
    else:
        subject01 = [associationcouple[0]]
    if associationcouple[1].find("/") >= 0:
        object01 = associationcouple[1].split("/")
    else:
        object01 = [associationcouple[1]]
    for s in subject01:
        for o in object01:
          couples.append((s, o))
  # Creating the Matrix for the Associations
  
  if couples != []:
    for q in qualifiers:
      row = []
      for r in qualifiers:
        prop = round(couples.count((q, r)) / NPub, 3)
        row.append(prop)
      matrix += row
    return matrix

#read df2 from pkl file
df2 = pd.read_pickle("results.pkl")

#print the first subject and object
with open('Id_Term.json', 'r') as fp:
    ID_T = json.load(fp)
model = BaselineModelSuperClasses().to(device)
model.load_state_dict(torch.load('models/new_data_super_classes_best_model_1.zip', map_location='cpu'))
model2 = BaselineModel().to(device)
model2.load_state_dict(torch.load('models/new_data_best_model_1.zip', map_location='cpu'))
superclasses = ["Taxonomic", "Symmetric", "Other", "Non-Symmetric", "Other"]
df_reltype = pd.read_csv("https://raw.githubusercontent.com/SisonkeBiotik-Africa/MeSH2Wikidata/main/new_encoding.csv")
superc_id = {"Taxonomic": ["P279", "P31", "P361"],
            "Symmetric": ["P769", "P2789", "P2293"],
            "Non-Symmetric": ["P1050", "P1057", "P1060", "P128", "P1349", "P1420", "P1582", "P1605", "P1606", "P171",
                        "P1909", "P1910", "P1916", "P1924", "P1995", "P2175", "P2176", "P2239", "P2597", "P2841",
                        "P2849", "P3094", "P3189", "P3190", "P3261", "P3262", "P3490", "P3491", "P3493", "P3781",
                        "P4545", "P4774", "P4954", "P5131", "P5132", "P5572", "P5642", "P636", "P680", "P681",
                        "P682", "P688", "P689", "P702", "P703", "P7500", "P780", "P8339", "P923", "P924", "P925",
                        "P926", "P927"],
            "Other": []}
def process(i):
  try:
      if(i%500==0):
        print("relation ",i)
      subj = ID_T[df2["subject"][i]]
      obj = ID_T[df2["object"][i]]
      inputd = torch.as_tensor(get_matrix(subj,obj))
      superclass_id = model(inputd).argmax().item()
      relation_type_id = model2(inputd).argmax().item()
      superclass = superclasses[superclass_id]
      relation_type = df_reltype["label"][relation_type_id]
      if not(relation_type in superc_id[superclass]):
          relation_type = "N/A"
      print("relation ",i)
      print("Subject:", subj, "\nProperty:", relation_type, "\nObject:", obj, "\nReferences:", PubIds[:3])
      df2["property"][i] = relation_type
      df2["references"][i] = PubIds[:3]
      #update the pkl file
      df2.to_pickle("results.pkl")
  except:
      return

pd.set_option('mode.chained_assignment', None)
Parallel(n_jobs=6, prefer="threads")(delayed(process)(i) for i in range(41121,200000))
