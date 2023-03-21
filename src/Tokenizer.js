const { BertWordPieceTokenizer } = require('@nlpjs/bert-tokenizer');

class Tokenizer extends BertWordPieceTokenizer {
    constructor(options = {}) {
        super({ settings: {} });
        this.lowercase = options.lowercase ?? true;
        this.configuration = {
            clsToken: '[CLS]',
            maskToken: '[MASK]',
            padToken: '[PAD]',
            sepToken: '[SEP]',
            unkToken: '[UNK]',
        };
        this.tokenPositions = {
            clsToken: 101,
            maskToken: 103,
            padToken: 0,
            sepToken: 102,
            unkToken: 100,
        };

        if (options.vocabContent) {
            this.loadDictionary(options.vocabContent);
        }
    }

    getBestPrefix(word) {
        const maxLength = Math.min(word.length - 1, this.affixMaxLength);
        for (let i = maxLength; i > 0; i -= 1) {
            const prefix = word.substring(0, i);
            if (this.words[prefix]) {
                return prefix;
            }
        }
    }

    getBestAffix(word) {
        const maxLength = Math.min(word.length - 1, this.affixMaxLength);
        for (let i = 1; i <= maxLength; i += 1) {
            const suffix = word.substring(word.length - i);
            if (this.words[suffix]) {
                return suffix;
            }
        }
    }

    tokenizeWithPositions(text) {
        const tokens = [];
        const positions = [];
        let position = 0;
        const words = text.trim().split(/\s+/);
        for (let i = 0; i < words.length; i += 1) {
            const word = words[i];
            if (this.lowercase) {
                word = word.toLowerCase();
            }
            let token = this.getBestPrefix(word) || this.getBestAffix(word);
            if (!token) {
                token = this.configuration.unkToken;
            }
            tokens.push(token);
            positions.push(position);
            position += token.length;
            if (i !== words.length - 1) {
                tokens.push(this.configuration.sepToken);
                positions.push(position);
                position += this.configuration.sepToken.length;
            }
        }
        tokens.unshift(this.configuration.clsToken);
        positions.unshift(0);
        tokens.push(this.configuration.sepToken);
        positions.push(position);
        return { tokens, positions };
    }
}

module.exports = Tokenizer;