import time, math
from library import parseDataset, getAttributes, getModelResponse, writeToFile

# https://figshare.com/articles/dataset/Seven_Phishing_Email_Datasets/25432108
# new dataset
def main():
    feats, labels = parseDataset("dataset", "Phishing_Email.csv")
    
    attribute = getAttributes()
    sys_ins = attribute["sys_ins"]
    max_temp = attribute["max_temp"]

    # https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash
    response_per_minute_limit = 15
    requests_per_day_limit = 1500 * 2
    
    personal_limit = 2000
    start = int(open("marker.txt", "r").read())
    threshold = start % response_per_minute_limit
    count = start

    try:
        while count < len(feats):
            prompt = feats[count]
            if count % response_per_minute_limit == threshold and count != start and count != personal_limit:
                time.sleep(60) # to avoid rate and response limits
            if count == requests_per_day_limit or count == personal_limit: # no more research for the day or personal testing
                break

            # if csv has an empty cell, convert to an empty string with a space(the model doesn't allow empty values)
            if isinstance(prompt, float) and math.isnan(prompt): 
                prompt = " "
            writeToFile(getModelResponse(sys_ins, prompt) + '\n')
            count += 1
    finally:
        with open("marker.txt", "w") as f:
            f.write(str(count))

if __name__ == "__main__":
    main()