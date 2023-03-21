const Tokenizer = require('./Tokenizer')

const onnxruntime = require('onnxruntime-web')

const fs = require('fs')

const vocab = fs.readFileSync('./resources/vocab.txt', 'utf-8');
const tokenizer = new Tokenizer({
    vocabContent: vocab,
    lowercase: true,
});
const ortSession = onnxruntime.InferenceSession.create('./resources/model.onnx');

const IntentClassifierScore = async(text) => {
    const startTime = Date.now();
    let tokens = tokenizer.encode(text).ids.map(v => BigInt(v));
    if (512 < tokens.length)
        tokens = tokens.slice(-512);

    const input = {
        input_ids: new onnxruntime.Tensor('int64', tokens, [1, tokens.length]),
        attention_mask: new onnxruntime.Tensor('int64', Array(tokens.length).fill(BigInt(1)), [1, tokens.length]),
        token_type_ids: new onnxruntime.Tensor('int64', Array(tokens.length).fill(BigInt(0)), [1, tokens.length])
    }
    const output = await (await ortSession).run(input);
    const endTime = Date.now();

    return {
        prediction: 1 / (1 + Math.exp(-Number(output.logits.data[1]))),
        latency: endTime - startTime
    };
}

const IsOfftopic = async(text) => {
    const result = await IntentClassifierScore(text);

    return {
        isOffTopic: 0.8 < result.prediction,
        latency: result.latency,
        score: result.prediction,
    }
}

const content = fs.readFileSync('src/main.js', 'utf-8');
IsOfftopic(content).then(result => {
    console.log('result', result);
}).catch(error => {
    console.log('error', error);
})

