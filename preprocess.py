# Load Dataset and preprocess

def preprocess_data(data, translator, word2vec, maxPassageLen=0, maxQuestionLen=0): 
  Xp = []
  Xq = []
  Ys = []
  Ye = []
  mapper = []
  for t in data:
    for paragraph in t['paragraphs']:
      #Convert context to word2vec
      paragraph_w2v_repr = []
      clean_paragraph = paragraph['context'].translate(translator).split(' ')
      if len(clean_paragraph) > maxPassageLen: maxPassageLen = len(clean_paragraph)
      for word in clean_paragraph:
        try: 
          word_w2v_repr = word2vec.wv.get_vector(word)
        except KeyError:
          word_w2v_repr = np.zeros((300))
        paragraph_w2v_repr.append(word_w2v_repr)
      Xp.append(paragraph_w2v_repr)
        
      #Convert question and answer to word2vec
      for qa in paragraph['qas']:
        #Convert question
        question_w2v_repr = []
        clean_qa = qa['question'].translate(translator).split(' ')
        
        if len(clean_qa) > maxQuestionLen and len(clean_qa) != 25601: 
          maxQuestionLen = len(clean_qa)
          
        else: 
          for word in clean_qa:
            try: 
              word_w2v_repr = word2vec.wv.get_vector(word)
            except KeyError:
              word_w2v_repr = np.zeros((300))
            question_w2v_repr.append(word_w2v_repr)
          #convert answer
          for answer in qa['answers']:
            clean_answer = answer['text'].translate(translator).split(' ')
            start_point_char = answer['answer_start']
            end_point_char = start_point_char + len(answer['text'])
            complete_answer = paragraph['context'][start_point_char : end_point_char].translate(translator).split(' ')
            indices = [i for i, x in enumerate(clean_paragraph) if x == complete_answer[-1]]
            for i in indices:
              if clean_paragraph[i - (len(complete_answer) - 1)] == complete_answer[0]:
                
                start_point = i - (len(complete_answer) - 1)
                end_point = i
                Xq.append(question_w2v_repr)
                Ys.append(start_point)
                Ye.append(end_point)
                mapper.append(t['paragraphs'].index(paragraph))
                break
  return np.array(Xp).T, np.array(Xq).T, to_categorical(Ys, num_classes=maxPassageLen), to_categorical(Ye, num_classes=maxPassageLen), mapper, maxPassageLen, maxQuestionLen