import preprocess, detection, results

def main():
    preprocess.createDataset()
    detection.detectionModel()
    #results.activationMap()
    #results.numSpeciesGraphic()
    
if __name__ == "__main__":
    main()