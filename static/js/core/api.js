export async function requestJson(url, body) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    let data = {};
    try {
        data = await response.json();
    } catch (error) {
        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }
    }

    if (!response.ok) {
        throw new Error(data.error || data.message || `Request failed with status ${response.status}`);
    }

    return data;
}
