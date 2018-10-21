#ifndef _IOFILE_H_
#define _IOFILE_H_


#include <string>
#include <vector>
//#include <minwinbase.h>
//#include "opencv2/opencv.hpp"
#define _S(str) ((str).c_str())
using std::string;
using std::vector;
//using cv::CMat;
typedef const string CStr;

typedef vector<string> vecS;
namespace unre
{

	struct FileOP
	{
	private:
		// Get file names from a wildcard. Eg: GetNames("D:\\*.jpg", imgNames);
		static int GetNames(CStr &nameW, vecS &names, string &dir);

		static int GetNames(CStr &nameW, vecS &names) { string dir; return GetNames(nameW, names, dir); };
		static int GetNames(CStr& rootFolder, CStr &fileW, vecS &names);
	public:

		static string BrowseFile(const char* strFilter = "Images (*.jpg;*.png)\0*.jpg;*.png\0All (*.*)\0*.*\0\0", bool isOpen = true);
		static string BrowseFolder();

		static inline string GetFolder(CStr& path);
		static inline string GetSubFolder(CStr& path);
		static inline string GetName(CStr& path);
		static inline string GetNameNE(CStr& path);
		static inline string GetPathNE(CStr& path);
		static inline string GetNameNoSuffix(CStr& path, CStr &suffix);

		static const string& GetCurrentDir();
		static int GetFullName(CStr& folder, vecS& subFolders);
		static int GetFullName(CStr& rootFolder, CStr &fileW, vecS &names);
		static int GetFullPath(CStr& rootFolder, CStr &fileW, vecS &names);
		static int GetNameNoSuffix(CStr& rootFolder, CStr &fileW, vecS &names);

		static inline string GetExtention(CStr name);


		//static bool FilesExist(CStr& fileW);
		static bool FileExist(CStr& filePath);
		static bool FilesExist(CStr& fileW);
		static bool FolderExist(CStr& strPath);

		static string GetWkDir();

		static bool MkDir(CStr&  _path);


		static int GetSubFolders(CStr& folder, vecS& subFolders);

	};
}
#endif // !_IOFILE_H_