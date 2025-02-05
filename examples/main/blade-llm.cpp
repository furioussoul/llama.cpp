#include <console.h>
#include "llama.h"
#include <log.h>
#include <sampling.h>

#include <cassert>
#include <iostream>

#include "arg.h"
#include "chat-template.hpp"
#include "common.h"
#include "llama.h"

#define UNUSED GGML_UNUSED

static const char *               DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant";

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    params.model               = argv[4];
    params.cpuparams.n_threads = 2;

    common_init_result llama_init = common_init_from_params(params);

    llama_model *   model = llama_init.model.get();
    llama_context * ctx   = llama_init.context.get();

    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab *          vocab          = llama_model_get_vocab(model);
    auto                         chat_templates = common_chat_templates_from_model(model, params.chat_template);
    std::vector<common_chat_msg> chat_msgs;

    auto chat_add_and_format = [&chat_msgs, &chat_templates, &params](const std::string & role,
                                                                      const std::string & content) {
        common_chat_msg new_msg{ role, content, {} };
        auto formatted = common_chat_format_single(*chat_templates.template_default, chat_msgs, new_msg, role == "user",
                                                   params.use_jinja);
        chat_msgs.push_back({ role, content, {} });
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };

    std::string prompt = chat_add_and_format("system", params.prompt.empty() ? DEFAULT_SYSTEM_MESSAGE : params.prompt);

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_inp;

    embd_inp = common_tokenize(ctx, prompt, true, true);

    std::string user_inp = chat_add_and_format("user", std::move("你好，鸡你太美～ 知道什么意思吗？"));
    const auto  line_inp = common_tokenize(ctx, user_inp, false, true);
    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

    auto sparams         = llama_sampler_chain_default_params();
    sparams.no_perf      = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const auto  t_main_start = ggml_time_us();
    int         n_decode     = 0;
    llama_token new_token_id;
    int         n_predict = 1024;

    llama_batch batch    = llama_batch_get_one(embd_inp.data(), embd_inp.size());

    for (int n_pos = 0; n_pos + batch.n_tokens < (int32_t)embd_inp.size() + n_predict;) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int  n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n", __func__, n_decode,
            (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);

    return 0;
}
