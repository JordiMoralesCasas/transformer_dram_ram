import h5py
from tqdm import tqdm

"""
    Functions copied from Google's "Prepare SVHN Dataset" Colab: https://colab.research.google.com/github/ozanpekmezci/capstone/blob/master/prepare-svhn.ipynb

    Modified to work without deprecated .value attribute.

"""
# The DigitStructFile is just a wrapper around the h5py data.  It basically references 
#    inf:              The input h5 matlab file
#    digitStructName   The h5 ref to all the file names
#    digitStructBbox   The h5 ref to all struc data
class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.dataset_split = inf.split("/")[-2]
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

    # getName returns the 'name' string for for the n(th) digitStruct. 
    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]][()]])

    # bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr[()][j].item()][()][0][0] for j in range(len(attr))]
        else:
            attr = [attr[()][0][0]]
        return attr

    # getBbox returns a dict of data for the n(th) bbox. 
    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

    # getAllDigitStructure returns all the digitStruct from the input file.     
    def getAllDigitStructure(self):
        all_digits = []
        for i in tqdm(range(len(self.digitStructName)), desc="Getting All Digit Structures..."):
            all_digits.append(self.getDigitStructure(i))
        return all_digits

    # Return a restructured version of the dataset (one structure by boxed digit).
    #
    #   Return a list of such dicts :
    #      'filename' : filename of the samples
    #      'boxes' : list of such dicts (one by digit) :
    #          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
    #          'left', 'top' : position of bounding box
    #          'width', 'height' : dimension of bounding box
    #
    # Note: We may turn this to a generator, if memory issues arise.
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 
                'filename' : pictDat[i]["name"],
                'dataset_split' : self.dataset_split
                }
            figures = []
            skip = False
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['height'] = pictDat[i]['height'][j]
                figure['label']  = pictDat[i]['label'][j]
                figure['left']   = pictDat[i]['left'][j]
                figure['top']    = pictDat[i]['top'][j]
                figure['width']  = pictDat[i]['width'][j]
                figures.append(figure)
                if figure['height'] < 0 or figure['left'] < 0 or figure['top'] < 0 or figure['width'] < 0:
                    skip = True
                    break
                
            if skip:
                # current sample is wrongfully formated
                continue
                
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result