#pragma once 

#include <string>
#include <sstream>

#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>

#include <nlohmann/json.hpp>

namespace btc {

class notifier {
public:

    notifier() = delete;

    static void notify(int id, const std::string& message) {
        std::string url("https://api.telegram.org/bot6564123355:AAGODWnjujnG4njAVSHuMwjyIUmzLVYG7kA/sendMessage");

        curlpp::Easy request;
        request.setOpt(curlpp::options::Url(url));

        nlohmann::json data;
        data["chat_id"] = id;
        data["text"] = message;

        std::string json_str = data.dump();

        request.setOpt(curlpp::options::PostFields(json_str));
        request.setOpt(curlpp::options::PostFieldSize(json_str.length()));
        request.setOpt<curlpp::options::HttpHeader>({"Content-Type: application/json"});

        std::stringstream ss;
        request.setOpt<curlpp::options::WriteStream>(&ss);

        request.perform();
    }

};

}
