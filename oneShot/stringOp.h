#ifndef _STRING_OP_H_
#define _STRING_OP_H_
#include <string>
#include <fstream>
#include "iofile.h"
namespace unre
{
	class StringOP
	{
	public:
		static std::string parseJsonFile2str(const char*jsonFilePath)
		{
			if (!FileOP::FilesExist(jsonFilePath))
			{
				return "";
			}
			else
			{
				std::fstream fi(jsonFilePath, std::ios::in);
				string stringFromStream;
				string line;
				while (std::getline(fi, line))
				{
					stringFromStream.append(line + "\n");
				}
				fi.close();
				return std::move(stringFromStream);
			}
		}
		static std::vector<std::string> splitString(const std::string& src, const std::string &symbols, bool repeat=false)
		{
			std::vector<std::string> result;
			int startIdx = 0;
			for (int i = 0; i<src.length(); i++)
			{
				bool isMatch = false;
				for (int j = 0; j<symbols.length(); j++)
				{
					if (src[i] == symbols[j])
					{
						isMatch = true;
						break;
					}
					if (!repeat)
					{
						break;
					}
				}
				if (isMatch)
				{
					std::string sub = src.substr(startIdx, i - startIdx);
					startIdx = i + 1;
					if (sub.length()>0)
					{
						result.push_back(sub);
					}
				}
				if (i + 1 == src.length())
				{
					std::string sub = src.substr(startIdx, src.length() - startIdx);
					startIdx = i + 1;
					if (sub.length()>0)
					{
						result.push_back(sub);
					}

				}
			}
			return std::move(result);
		}
	};


}
#endif // !_STRING_OP_H_
