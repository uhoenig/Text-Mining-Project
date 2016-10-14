#Text Mining Script:
#Check the "OnlineNewsPopularity.txt" file for a detailed overview of the data 
setwd("/Users/uhoenig/Dropbox/Studium 2015-16 GSE/Term 3/Text Mining/Project/Data")

library(dplyr)
raw_train <- tbl_df(read.csv(file="news_popularity_training.csv", stringsAsFactors = F))

table(raw_train$popularity)
#select viral links: 3+4+5

url_viral=filter(raw_train, popularity %in% c(3,4,5))
url_nonviral=filter(raw_train, popularity %in% c(1,2))
pop_viral=data.frame(link = url_viral$url)
pop_nonviral=data.frame(link = url_nonviral$url)

write.csv(pop_viral, file = "pop5.csv", row.names = FALSE)
write.csv(pop_nonviral, file = "pop1.csv", row.names = FALSE)

##Analysis starts here

Needed <- c("tm", "SnowballCC", "RColorBrewer", "ggplot2", "wordcloud", "biclust", "cluster", "igraph", "fpc")   
install.packages(Needed, dependencies=TRUE)   
install.packages("Rcampdf", repos = "http://datacube.wu.ac.at/", type = "source")    

library(tm)   
Documents_viral_clean_R <- read.csv("Documents_viral_clean_R.txt", header=FALSE, stringsAsFactors=FALSE)$V1
Documents_nonviral_clean_R <- read.csv("Documents_nonviral_clean_R.txt", header=FALSE, stringsAsFactors=FALSE)$V1

corp_viral <- Corpus(VectorSource(Documents_viral_clean_R))
corp_nonviral <- Corpus(VectorSource(Documents_nonviral_clean_R))
#Document-Term Matrix
dtm_viral <- DocumentTermMatrix(corp_viral)
dtm_nonviral <- DocumentTermMatrix(corp_nonviral)

# -----VIRAL FIRST-----

### Explore your data      
freq <- colSums(as.matrix(dtm_viral))   
length(freq)   
ord <- order(freq)   
m <- as.matrix(dtm_viral)   
dim(m)   

#  Start by removing sparse terms:   
dtms <- removeSparseTerms(dtm_viral, 0.9) # This makes a matrix that is 80% empty space, maximum.
### Word Frequency   
head(table(freq), 20) 
# The above output is two rows of numbers. The top number is the frequency with which 
# words appear and the bottom number reflects how many words appear that frequently. 
#
tail(table(freq), 20)
# **View a table of the terms after removing sparse terms, as above.
freq <- colSums(as.matrix(dtms))   
freq 
# The above matrix was created using a data transformation we made earlier. 
# **An alternate view of term frequency:**   
# This will identify all terms that appear frequently (in this case, 50 or more times).   
findFreqTerms(dtm_viral, lowfreq=1000)   # Change "50" to whatever is most appropriate for your data.

### Plot Word Frequencies
# **Plot words that appear at least 50 times.**   
library(ggplot2)   
wf <- data.frame(word=names(freq), freq=freq)
wf$word <- factor(wf$word, levels = wf$word[order(wf$freq, decreasing = TRUE)])
p <- ggplot(subset(wf, freq>1500), aes(word, freq))    
p <- p + geom_bar(stat="identity")   
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))   
p   

### Word Clouds!   
# First load the package that makes word clouds in R.    
library(wordcloud)   
dtms <- removeSparseTerms(dtm_viral, 0.65) # Prepare the data (max 15% empty space)   
freq <- colSums(as.matrix(dtm_viral)) # Find word frequencies   
dark2 <- brewer.pal(6, "Dark2")   
wordcloud(names(freq), freq, max.words=30, colors=dark2)

tfidfviral=read.csv("tfidfviral.csv")
tail(tfidfviral)
pal2 <- brewer.pal(8,"Dark2")
frame <- tfidfviral
frame$freq=round(frame$freq*100,0)
wordcloud(frame$word[1:30],freq=frame$freq[1:30], scale=c(1,6), colors = pal2)


frame1 <- tfidf1k_1[order(tfidf1k_1$num),]

wordcloud(frame1$name[1:50],frame1$num[1:50],scale=c(1,1), colors = pal2)

