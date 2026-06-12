import preprocess, detection

def main():
    preprocess.createDataset()
    detection.detectionModel()
    
if __name__ == "__main__":
    main()