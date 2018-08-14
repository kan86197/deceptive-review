library(keras)

my_data <- read.csv(file = "C:\\Users\\USER\\Desktop\\Why\\deceptive_opinion_shuffled.csv", header = TRUE, sep = ",")

labels <- as.numeric(my_data$deceptive == "truthful")
texts <- my_data$text
labels <- as.character(labels)
texts <- as.character(texts)

maxlen <- 100
training_samples <- 800
validation_samples <- 200
testing_samples <- 200
max_words <- 10000
tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index

cat("Found", length(word_index), "unique tokens.\n")
data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "\n")
cat('Shape of label tensor:', dim(labels), "\n")
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):
                                (training_samples + validation_samples)]
test_indices <- indices[(training_samples + validation_samples + 1):
                          (training_samples + validation_samples + testing_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]
x_test <- data[test_indices,]
y_test <- labels[test_indices]


glove_dir = "C:\\Users\\USER\\Downloads\\glove.6B"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}
cat("Found", length(embeddings_index), "word vectors.\n")

embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      embedding_matrix[index+1,] <- embedding_vector
  }
}

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 32) %>%
  layer_lstm(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 7,
  batch_size = 50,
  validation_data = list(x_val, y_val)
)

plottedGraph <- plot(history)
plottedGraph

save_model_weights_hdf5(model, "deceptive_review.h5")

model %>% 
  load_model_weights_hdf5("deceptive_review.h5") %>% 
  evaluate(x_test, y_test)

